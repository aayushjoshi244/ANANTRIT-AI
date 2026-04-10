import os
import time
import cv2

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
    telegram_token="your_token",
    telegram_chat_id="your_chat_id",
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

def draw_guard_overlay(frame):
    name, box, confidence = guard.get_overlay()

    if name is None or box is None:
        return frame

    x, y, w, h = box

    if name == "Unknown":
        color = (0, 0, 255)
        label = "Unknown"
    else:
        color = (0, 255, 0)
        label = str(name)

    if confidence is not None:
        label = f"{label} ({confidence:.1f})"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text_y = max(30, y - 10)
    text_width = max(180, len(label) * 12)
    cv2.rectangle(frame, (x, text_y - 25), (x + text_width, text_y + 5), color, -1)

    cv2.putText(
        frame,
        label,
        (x + 5, text_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def mjpeg_generator():
    while True:
        frame = camera.get_frame()

        if frame is None:
            time.sleep(0.05)
            continue

        frame = frame.copy()
        frame = draw_guard_overlay(frame)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
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
