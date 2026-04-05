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
    def __init__(
        self,
        state: StateStore,
        camera_manager,
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

        self.motion_area_threshold = 5000
        self.save_cooldown_seconds = 5
        self.unknown_distance_threshold = 3000.0
        self.last_save_time = 0.0
        self.last_welcome_time: dict[str, float] = {}
        self.welcome_cooldown_seconds = 8

        self.prev_frame: Optional[np.ndarray] = None

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_detector.empty():
            raise RuntimeError("Could not load Haar cascade from OpenCV data path")

        self.knn = self._load_face_model()

        os.makedirs(self.capture_dir, exist_ok=True)

    def speak(self, text: str) -> None:
        try:
            subprocess.run(["espeak", text], check=False)
        except FileNotFoundError:
            print("espeak not installed")

    def _load_face_model(self) -> KNeighborsClassifier:
        if not os.path.isfile(self.names_path):
            raise FileNotFoundError(f"Missing file: {self.names_path}")
        if not os.path.isfile(self.faces_path):
            raise FileNotFoundError(f"Missing file: {self.faces_path}")

        with open(self.names_path, "rb") as f:
            labels = pickle.load(f)
        with open(self.faces_path, "rb") as f:
            faces = pickle.load(f)

        faces = np.asarray(faces)
        labels = np.asarray(labels)

        if len(faces) == 0 or len(labels) == 0:
            raise ValueError("Training data is empty")
        if len(faces) != len(labels):
            raise ValueError("faces_data.pkl and names.pkl lengths do not match")

        knn = KNeighborsClassifier(n_neighbors=5)
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

    def _motion_detected(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        return any(
            cv2.contourArea(contour) >= self.motion_area_threshold
            for contour in contours
        )

    def _should_welcome(self, name: str) -> bool:
        now = time.time()
        last = self.last_welcome_time.get(name, 0.0)
        if now - last >= self.welcome_cooldown_seconds:
            self.last_welcome_time[name] = now
            return True
        return False

    def _loop(self) -> None:
        while self.running:
            frame = self.camera_manager.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            if self.prev_frame is None:
                self.prev_frame = frame
                time.sleep(0.05)
                continue

            frame1 = self.prev_frame
            frame2 = frame.copy()
            self.prev_frame = frame2

            motion_detected = self._motion_detected(frame1, frame2)
            if not motion_detected:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            if len(faces) == 0:
                self.state.mark_event("Motion detected")
                time.sleep(0.08)
                continue

            for (x, y, w, h) in faces:
                crop_img = frame1[y:y + h, x:x + w]
                if crop_img.size == 0:
                    continue

                try:
                    resized = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                except Exception:
                    continue

                prediction = self.knn.predict(resized)[0]
                distances, _ = self.knn.kneighbors(resized, n_neighbors=1)
                distance = float(distances[0][0])

                if distance > self.unknown_distance_threshold:
                    name = "Unknown"
                else:
                    name = str(prediction)

                now = time.time()
                if now - self.last_save_time < self.save_cooldown_seconds:
                    continue

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(self.capture_dir, f"{name}_{timestamp}.jpg")
                cv2.imwrite(filename, frame1)

                if name == "Unknown":
                    self.speak("Unknown person detected")
                    self._send_telegram(filename, "Unknown person detected")
                    self.state.mark_event("Unknown person detected")
                else:
                    if self._should_welcome(name):
                        self.speak(f"Welcome {name}")
                    self.state.mark_event(f"Recognized {name}")

                self.last_save_time = now

            time.sleep(0.08)