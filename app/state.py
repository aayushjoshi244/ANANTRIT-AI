from dataclasses import dataclass
import time


@dataclass
class AppState:
    guard_enabled: bool = False
    camera_online: bool = False
    assistant_online: bool = False
    last_command: str | None = None
    last_event: str | None = None
    last_event_time: float = 0.0
    telegram_enabled: bool = True
    stream_clients: int = 0


class StateStore:
    def __init__(self) -> None:
        self._state = AppState()

    def get(self) -> AppState:
        return self._state

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self._state, k, v)

    def mark_event(self, message: str) -> None:
        self._state.last_event = message
        self._state.last_event_time = time.time()