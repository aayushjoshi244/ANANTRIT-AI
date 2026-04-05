import os
import subprocess
from typing import Callable
from app.state import StateStore


def speak(text: str) -> None:
    subprocess.run(["espeak", text], check=False)


class CommandRouter:
    def __init__(
        self,
        state: StateStore,
        on_guard_start: Callable[[], None],
        on_guard_stop: Callable[[], None],
    ) -> None:
        self.state = state
        self.on_guard_start = on_guard_start
        self.on_guard_stop = on_guard_stop

    def execute(self, raw_command: str) -> str:
        command = raw_command.strip().lower()
        self.state.update(last_command=command)

        if command == "start guard":
            self.on_guard_start()
            speak("Guard mode started")
            return "Guard mode started"

        if command == "stop guard":
            self.on_guard_stop()
            speak("Guard mode stopped")
            return "Guard mode stopped"

        if command == "status":
            s = self.state.get()
            return (
                f"Guard is {'on' if s.guard_enabled else 'off'}, "
                f"camera is {'online' if s.camera_online else 'offline'}"
            )

        if command == "take snapshot":
            return "Snapshot feature will be connected in the next step"

        if command == "shutdown":
            speak("Shutdown command received")
            # Keep disabled for now, safer during development
            return "Shutdown blocked in development mode"

        if command == "reboot":
            speak("Reboot command received")
            return "Reboot blocked in development mode"

        return f"Unknown command: {command}"