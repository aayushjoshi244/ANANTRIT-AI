from pydantic import BaseModel, Field
from typing import Optional


class StatusResponse(BaseModel):
    guard_enabled: bool
    camera_online: bool
    assistant_online: bool
    last_command: Optional[str]
    last_event: Optional[str]
    last_event_time: Optional[float]
    telegram_enabled: bool
    stream_clients: int


class CommandRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=100)


class ActionResponse(BaseModel):
    ok: bool
    message: str


class GuardToggleResponse(BaseModel):
    ok: bool
    guard_enabled: bool
    message: str