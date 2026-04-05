import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import (
    StatusResponse,
    CommandRequest,
    ActionResponse,
    GuardToggleResponse,
    AlertsResponse,
    AlertItem,
)
from app.state import StateStore
from app.guard_service import GuardService
from app.camera_manager import CameraManager
from app.command_router import CommandRouter
from app.alerts_store import AlertsStore

app = FastAPI(title="Anantrit Core", version="0.1.0")

state = StateStore()
alerts_store = AlertsStore()
camera = CameraManager(state=state, width=640, height=480)

guard = GuardService(
    state=state,
    camera_manager=camera,
    alerts_store=alerts_store,
    telegram_token="8598505116:AAHsH_uPijhH7L_3_ELqqfRFXBsNCHjpO40",
    telegram_chat_id="1678217255",
)

router = CommandRouter(
    state=state,
    on_guard_start=guard.start,
    on_guard_stop=guard.stop,
)


@app.on_event("startup")
def startup_event() -> None:
    state.update(assistant_online=True)

    try:
        camera.start()
        print("Camera started successfully")
        state.mark_event("Camera started")
    except Exception as e:
        print("Camera startup error:", e)
        state.mark_event(f"Camera failed: {e}")


@app.on_event("shutdown")
def shutdown_event() -> None:
    guard.stop()
    camera.stop()
    state.update(assistant_online=False)
    state.mark_event("Anantrit backend stopped")


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    s = state.get()
    return StatusResponse(**s.__dict__)


@app.post("/command", response_model=ActionResponse)
def run_command(payload: CommandRequest) -> ActionResponse:
    result = router.execute(payload.command)
    return ActionResponse(ok=True, message=result)


@app.post("/guard/start", response_model=GuardToggleResponse)
def start_guard() -> GuardToggleResponse:
    guard.start()
    return GuardToggleResponse(ok=True, guard_enabled=True, message="Guard started")


@app.post("/guard/stop", response_model=GuardToggleResponse)
def stop_guard() -> GuardToggleResponse:
    guard.stop()
    return GuardToggleResponse(ok=True, guard_enabled=False, message="Guard stopped")


@app.get("/alerts", response_model=AlertsResponse)
def get_alerts():
    items = alerts_store.list_alerts(limit=50)
    return AlertsResponse(
        ok=True,
        alerts=[AlertItem(**item) for item in items],
    )


@app.get("/")
def root():
    return JSONResponse(
        {
            "name": "Anantrit Core",
            "version": "0.1.0",
            "docs": "/docs",
            "status": "/status",
            "stream": "/stream",
            "snapshot": "/snapshot",
            "alerts": "/alerts",
        }
    )


def mjpeg_generator():
    while True:
        frame = camera.get_jpeg_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        time.sleep(0.03)


@app.get("/stream")
def video_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
def snapshot():
    os.makedirs("captures", exist_ok=True)
    filename = time.strftime("captures/snapshot_%Y%m%d_%H%M%S.jpg")

    if camera.save_snapshot(filename):
        return {"ok": True, "path": filename}

    return {"ok": False, "message": "Could not capture snapshot"}


@app.websocket("/ws/status")
async def websocket_status(ws: WebSocket):
    await ws.accept()
    current = state.get()
    state.update(stream_clients=current.stream_clients + 1)

    try:
        while True:
            s = state.get()
            await ws.send_json(s.__dict__)
            await ws.receive_text()
    except WebSocketDisconnect:
        current = state.get()
        state.update(stream_clients=max(0, current.stream_clients - 1))
