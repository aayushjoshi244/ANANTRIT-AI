from pydantic import BaseModel
from typing import Optional


class StatusResponse(BaseModel):
    guard_enabled: bool
    camera_online: bool
    assistant_online: bool
    last_command: Optional[str]
    last_event: Optional[str]
    last_event_time: float
    telegram_enabled: bool
    stream_clients: int


class CommandRequest(BaseModel):
    command: str


class ActionResponse(BaseModel):
    ok: bool
    message: str


class GuardToggleResponse(BaseModel):
    ok: bool
    guard_enabled: bool
    message: str

class AlertItem(BaseModel):
    id: int
    event_type: str
    person_name: Optional[str]
    image_path: Optional[str]
    message: Optional[str]
    created_at: float


class AlertsResponse(BaseModel):
    ok: bool
    alerts: list[AlertItem]