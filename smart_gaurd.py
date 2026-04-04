import cv2
import time
import os
import pickle
import numpy as np
import subprocess
from sklearn.neighbors import KNeighborsClassifier


CAPTURE_DIR = "captures"
DATA_DIR = "data"
CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
NAMES_PATH = os.path.join(DATA_DIR, "names.pkl")
FACES_PATH = os.path.join(DATA_DIR, "faces_data.pkl")

MOTION_AREA_THRESHOLD = 5000
SAVE_COOLDOWN_SECONDS = 5
UNKNOWN_DISTANCE_THRESHOLD = 3000.0  # adjust later if needed


def speak(text: str) -> None:
    try:
        subprocess.run(["espeak", text], check=False)
    except FileNotFoundError:
        print("espeak not installed. Run: sudo apt install espeak -y")


def load_face_model():
    if not os.path.isfile(NAMES_PATH):
        raise FileNotFoundError(f"Missing file: {NAMES_PATH}")
    if not os.path.isfile(FACES_PATH):
        raise FileNotFoundError(f"Missing file: {FACES_PATH}")

    with open(NAMES_PATH, "rb") as f:
        labels = pickle.load(f)
    with open(FACES_PATH, "rb") as f:
        faces = pickle.load(f)

    faces = np.asarray(faces)
    labels = np.asarray(labels)

    if len(faces) == 0 or len(labels) == 0:
        raise ValueError("Training data is empty.")
    if len(faces) != len(labels):
        raise ValueError("faces_data.pkl and names.pkl lengths do not match.")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    return knn


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    if not os.path.isfile(CASCADE_PATH):
        print(f"Error: Missing Haar cascade file: {CASCADE_PATH}")
        return

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        print("Error: Could not load Haar cascade.")
        return

    try:
        knn = load_face_model()
    except Exception as e:
        print(f"Error loading face model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if not ret or not ret2:
        print("Error: Could not read initial frames from camera.")
        cap.release()
        return

    last_save_time = 0

    while True:
        diff = cv2.absdiff(frame1, frame2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = any(cv2.contourArea(c) >= MOTION_AREA_THRESHOLD for c in contours)

        display_frame = frame1.copy()

        if motion_detected:
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                crop_img = frame1[y:y+h, x:x+w]
                resized = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

                prediction = knn.predict(resized)[0]
                distances, _ = knn.kneighbors(resized, n_neighbors=1)
                distance = float(distances[0][0])

                if distance > UNKNOWN_DISTANCE_THRESHOLD:
                    name = "Unknown"
                    box_color = (0, 0, 255)
                else:
                    name = str(prediction)
                    box_color = (0, 255, 0)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.rectangle(display_frame, (x, y - 35), (x + w, y), box_color, -1)
                cv2.putText(
                    display_frame,
                    name,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                now = time.time()
                if now - last_save_time >= SAVE_COOLDOWN_SECONDS:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join(CAPTURE_DIR, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame1)
                    print(f"Saved: {filename}")

                    if name == "Unknown":
                        speak("Unknown person detected")
                    else:
                        speak(f"Welcome {name}")

                    last_save_time = now

        if motion_detected:
            cv2.putText(
                display_frame,
                "MOTION DETECTED",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Smart Guard", display_frame)

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()