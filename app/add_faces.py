import cv2
import pickle
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path("data")
CASCADE_PATH = DATA_DIR / "haarcascade_frontalface_default.xml"
FACES_PATH = DATA_DIR / "faces_data.pkl"
NAMES_PATH = DATA_DIR / "names.pkl"

MAX_SAMPLES = 120
SAVE_EVERY_N_FRAMES = 3
FACE_SIZE = (64, 64)

def preprocess_face(frame, x, y, w, h, output_size=FACE_SIZE):
    h_img, w_img = frame.shape[:2]

    pad = int(0.15 * max(w, h))
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
        gray = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    except Exception:
        return None

    gray = gray.astype(np.float32) / 255.0
    return gray.flatten()

def main():
    if not CASCADE_PATH.exists():
        raise FileNotFoundError(f"Missing cascade: {CASCADE_PATH}")

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade")

    name = input("Enter your name: ").strip()
    if not name:
        raise ValueError("Name cannot be empty")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    samples = []
    frame_count = 0

    print("Look at the camera with small pose changes and expressions. Press q to stop.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(80, 80)
        )

        # use largest face only
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            processed = preprocess_face(frame, x, y, w, h)

            if processed is not None and frame_count % SAVE_EVERY_N_FRAMES == 0 and len(samples) < MAX_SAMPLES:
                samples.append(processed)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Samples: {len(samples)}/{MAX_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Register Face", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(samples) >= MAX_SAMPLES:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 20:
        raise RuntimeError("Too few good face samples collected. Try again with better lighting.")

    faces_data = np.asarray(samples, dtype=np.float32)
    names = [name] * len(faces_data)

    os.makedirs(DATA_DIR, exist_ok=True)

    if NAMES_PATH.exists():
        with open(NAMES_PATH, "rb") as f:
            old_names = pickle.load(f)
    else:
        old_names = []

    if FACES_PATH.exists():
        with open(FACES_PATH, "rb") as f:
            old_faces = pickle.load(f)
        old_faces = np.asarray(old_faces, dtype=np.float32)
        all_faces = np.vstack([old_faces, faces_data])
    else:
        all_faces = faces_data

    all_names = old_names + names

    with open(FACES_PATH, "wb") as f:
        pickle.dump(all_faces, f)

    with open(NAMES_PATH, "wb") as f:
        pickle.dump(all_names, f)

    print(f"Saved {len(faces_data)} samples for {name}")

if __name__ == "__main__":
    main()