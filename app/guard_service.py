import os
import time
import pickle
import threading
import subprocess
from typing import Optional

import cv2
import numpy as np
import requests
from sklearn.neighbors import KNeighborsClassifier

from app.state import StateStore


class GuardService:
    FACE_SIZE = (64, 64)

    def __init__(
        self,
        state: StateStore,
        camera_manager,
        alerts_store,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self.state = state
        self.camera_manager = camera_manager
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.capture_dir = "captures"
        self.data_dir = "data"
        self.names_path = os.path.join(self.data_dir, "names.pkl")
        self.faces_path = os.path.join(self.data_dir, "faces_data.pkl")

        self.motion_area_threshold = 1800
        self.save_cooldown_seconds = 5
        self.unknown_distance_threshold = 0.85
        self.last_save_time = 0.0
        self.last_welcome_time: dict[str, float] = {}
        self.welcome_cooldown_seconds = 8
        self.alerts_store = alerts_store

        self.prev_motion_frame: Optional[np.ndarray] = None
        self.frame_skip = 2
        self.frame_counter = 0

        cascade_candidates = [
            os.path.join(self.data_dir, "haarcascade_frontalface_default.xml"),
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]

        cascade_path = None
        for path in cascade_candidates:
            if os.path.isfile(path):
                cascade_path = path
                break

        if cascade_path is None:
            raise FileNotFoundError(
                "Could not find haarcascade_frontalface_default.xml in data/ or system OpenCV paths"
            )

        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

        self.knn = self._load_face_model()
        os.makedirs(self.capture_dir, exist_ok=True)

    def speak(self, text: str) -> None:
        try:
            subprocess.run(["espeak", text], check=False)
        except FileNotFoundError:
            print("espeak not installed")

    def _preprocess_face(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        h_img, w_img = frame.shape[:2]
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            gray = cv2.resize(gray, self.FACE_SIZE, interpolation=cv2.INTER_AREA)
        except Exception:
            return None

        vec = gray.astype(np.float32) / 255.0
        return vec.flatten().reshape(1, -1)

    def _load_face_model(self) -> KNeighborsClassifier:
        if not os.path.isfile(self.names_path):
            raise FileNotFoundError(f"Missing file: {self.names_path}")
        if not os.path.isfile(self.faces_path):
            raise FileNotFoundError(f"Missing file: {self.faces_path}")

        with open(self.names_path, "rb") as f:
            labels = pickle.load(f)
        with open(self.faces_path, "rb") as f:
            faces = pickle.load(f)

        faces = np.asarray(faces, dtype=np.float32)
        labels = np.asarray(labels)

        if len(faces) == 0 or len(labels) == 0:
            raise ValueError("Training data is empty")
        if len(faces) != len(labels):
            raise ValueError("faces_data.pkl and names.pkl lengths do not match")

        n_neighbors = min(3, len(faces))
        if n_neighbors < 1:
            raise ValueError("Not enough training data")

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        knn.fit(faces, labels)
        return knn

    def _send_telegram(self, photo_path: str, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
        try:
            with open(photo_path, "rb") as photo:
                response = requests.post(
                    url,
                    data={"chat_id": self.telegram_chat_id, "caption": message},
                    files={"photo": photo},
                    timeout=15,
                )
            print("Telegram status:", response.status_code)
            print("Telegram response:", response.text)
        except Exception as e:
            print("Telegram error:", e)
            self.state.mark_event(f"Telegram error: {e}")

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.state.update(guard_enabled=True)
        self.state.mark_event("Guard started")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.state.update(guard_enabled=False)
        self.state.mark_event("Guard stopped")

    def _make_motion_frame(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _motion_detected(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        diff = cv2.absdiff(prev_frame, curr_frame)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return any(cv2.contourArea(contour) >= self.motion_area_threshold for contour in contours)

    def _should_welcome(self, name: str) -> bool:
        now = time.time()
        last = self.last_welcome_time.get(name, 0.0)
        if now - last >= self.welcome_cooldown_seconds:
            self.last_welcome_time[name] = now
            return True
        return False

    def _recognize_face(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> str:
        vec = self._preprocess_face(frame, x, y, w, h)
        if vec is None:
            return "Unknown"

        prediction = self.knn.predict(vec)[0]
        distances, indices = self.knn.kneighbors(vec, n_neighbors=min(3, self.knn.n_neighbors))
        mean_distance = float(np.mean(distances[0]))

        if mean_distance > self.unknown_distance_threshold:
            return "Unknown"

        neighbor_labels = [self.knn._y[idx] for idx in indices[0]]
        same_count = sum(1 for lbl in neighbor_labels if lbl == prediction)

        if same_count < 2 and len(neighbor_labels) >= 2:
            return "Unknown"

        return str(prediction)

    def _loop(self) -> None:
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                time.sleep(0.01)
                continue

            motion_frame = self._make_motion_frame(frame)

            if self.prev_motion_frame is None:
                self.prev_motion_frame = motion_frame
                time.sleep(0.03)
                continue

            motion_detected = self._motion_detected(self.prev_motion_frame, motion_frame)
            self.prev_motion_frame = motion_frame

            if not motion_detected:
                time.sleep(0.03)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                self.state.mark_event("Motion detected")
                time.sleep(0.05)
                continue

            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            for (x, y, w, h) in faces[:1]:
                name = self._recognize_face(frame, x, y, w, h)

                now = time.time()
                if now - self.last_save_time < self.save_cooldown_seconds:
                    continue

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(self.capture_dir, f"{name}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)

                if name == "Unknown":
                    self.speak("Unknown person detected")
                    self._send_telegram(filename, "Unknown person detected")
                    self.state.mark_event("Unknown person detected")
                else:
                    if self._should_welcome(name):
                        self.speak(f"Welcome {name}")
                    self.state.mark_event(f"Recognized {name}")

                self.last_save_time = now

            time.sleep(0.05)