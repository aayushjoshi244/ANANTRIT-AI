import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.state import StateStore


class CameraManager:
    def __init__(
        self,
        state: StateStore,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.state = state
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.state.update(camera_online=False)
            raise RuntimeError("Could not open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        self.state.update(camera_online=True)

    def _reader_loop(self) -> None:
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_jpeg_frame(self) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None

        return buffer.tobytes()

    def save_snapshot(self, path: str) -> bool:
        frame = self.get_frame()
        if frame is None:
            return False
        return cv2.imwrite(path, frame)

    def stop(self) -> None:
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.state.update(camera_online=False)