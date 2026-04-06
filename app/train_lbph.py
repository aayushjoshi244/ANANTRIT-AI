import cv2
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "app" / "data"
CASCADE_PATH = DATA_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = DATA_DIR / "face_model_lbph.yml"
LABELS_PATH = DATA_DIR / "face_labels.pkl"

MAX_SAMPLES = 120
SAVE_EVERY_N_FRAMES = 3
FACE_SIZE = (160, 160)


def open_camera():
    candidates = [
        lambda: cv2.VideoCapture(0),
        lambda: cv2.VideoCapture(0, cv2.CAP_V4L2),
    ]

    for make_cap in candidates:
        cap = make_cap()
        if cap is not None and cap.isOpened():
            return cap

    raise RuntimeError("Could not open camera. Stop the backend first, then try again.")


def extract_face_gray(frame, x, y, w, h):
    h_img, w_img = frame.shape[:2]
    pad = int(0.18 * max(w, h))

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    try:
        gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    except Exception:
        return None

    return gray


def save_labels(labels_map):
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels_map, f)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CASCADE_PATH.exists():
        raise FileNotFoundError(f"Missing cascade: {CASCADE_PATH}")

    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face is not available. Install opencv-contrib-python.")

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade")

    person_name = input("Enter your name: ").strip()
    if not person_name:
        raise ValueError("Name cannot be empty")

    labels_map = {person_name: 0}
    person_id = 0

    cap = open_camera()

    faces_list = []
    ids_list = []
    frame_count = 0

    print("Capture started. Move slightly left/right/up/down. Press q to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray_full,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(80, 80)
            )

            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            if faces:
                x, y, w, h = faces[0]
                face_gray = extract_face_gray(frame, x, y, w, h)

                if (
                    face_gray is not None
                    and frame_count % SAVE_EVERY_N_FRAMES == 0
                    and len(faces_list) < MAX_SAMPLES
                ):
                    faces_list.append(face_gray)
                    ids_list.append(person_id)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Samples: {len(faces_list)}/{MAX_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow("LBPH Face Registration", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or len(faces_list) >= MAX_SAMPLES:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(faces_list) < 25:
        raise RuntimeError("Too few samples captured. Try again in better lighting.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    recognizer.train(faces_list, np.array(ids_list, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    save_labels(labels_map)

    print(f"Saved LBPH model for {person_name} with label {person_id}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Labels path: {LABELS_PATH}")


if __name__ == "__main__":
    main()