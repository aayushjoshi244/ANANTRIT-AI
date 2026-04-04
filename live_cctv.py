from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

lock = threading.Lock()

def generate_frames():
    while True:
        with lock:
            success, frame = camera.read()

        if not success:
            continue

        ret,buffer = cv2.imencode(".jpeg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.route("/")
def index():
    return """
    <html>
        <head><title>Raspberry Pi Live CCTV</title></head>
        <body style="text-align:center; background:#111; color:white;">
            <h1>Raspberry Pi Live CCTV</h1>
            <img src="/video_feed" width="720">
        </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        camera.release()