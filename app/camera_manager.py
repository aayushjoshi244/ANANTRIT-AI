import threading
import time
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2

from app.state import StateStore


class CameraManager:
    def __init__(self, state: StateStore, width: int = 640, height: int = 480):
        self.state = state
        self.width = width
        self.height = height

        self.picam2: Optional[Picamera2] = None
        self.frame: Optional[np.ndarray] = None

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def _create_camera(self):
        for i in range(5):
            try:
                cams = Picamera2.global_camera_info()
                print("Camera list:", cams)

                if not cams:
                    raise RuntimeError("No camera detected")

                return Picamera2()

            except Exception as e:
                print("Retry camera:", e)
                time.sleep(1)

        raise RuntimeError("Camera not available")

    def start(self):
        if self.running:
            return

        try:
            self.picam2 = self._create_camera()

            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )

            self.picam2.configure(config)
            self.picam2.start()

            time.sleep(1)

            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with self.lock:
                self.frame = frame

            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

            self.state.update(camera_online=True)
            print("Camera started")

        except Exception as e:
            self.state.update(camera_online=False)
            print("Camera error:", e)

    def _loop(self):
        while self.running:
            try:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.frame = frame

            except Exception as e:
                print("Loop error:", e)
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_jpeg_frame(self):
        frame = self.get_frame()
        if frame is None:
            return None

        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def save_snapshot(self, path: str):
        frame = self.get_frame()
        if frame is None:
            return False
        return cv2.imwrite(path, frame)

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.picam2:
            self.picam2.stop()

        self.state.update(camera_online=False)