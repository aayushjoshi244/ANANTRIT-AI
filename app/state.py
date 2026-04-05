from dataclasses import dataclass, field
from threading import Lock
from typing import Optional
import time


@dataclass
class SystemState:
    guard_enabled: bool = False
    camera_online: bool = False
    assistant_online: bool = False
    last_command: Optional[str] = None
    last_event: Optional[str] = None
    last_event_time: Optional[float] = None
    telegram_enabled: bool = True
    stream_clients: int = 0


class StateStore:
    def __init__(self) -> None:
        self._state = SystemState()
        self._lock = Lock()

    def get(self) -> SystemState:
        with self._lock:
            return SystemState(**self._state.__dict__)

    def update(self, **kwargs) -> SystemState:
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._state, key):
                    raise AttributeError(f"Invalid state field: {key}")
                setattr(self._state, key, value)
            return SystemState(**self._state.__dict__)

    def mark_event(self, event: str) -> SystemState:
        with self._lock:
            self._state.last_event = event
            self._state.last_event_time = time.time()
            return SystemState(**self._state.__dict__)