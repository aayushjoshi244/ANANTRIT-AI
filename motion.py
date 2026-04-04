import cv2
import time
import os

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Create folder if not exists
os.makedirs("captures", exist_ok=True)

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        
        motion_detected = True

    if motion_detected:
        print("Motion detected!")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"captures/img_{timestamp}.jpg"

        cv2.imwrite(filename, frame1)
        print(f"Image saved: {filename}")

        time.sleep(2)  # avoid too many images

    cv2.imshow("Feed", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()