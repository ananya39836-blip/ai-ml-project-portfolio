import cv2
import torch
from gtts import gTTS
from playsound import playsound
import os
import threading
import time

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Webcam
cap = cv2.VideoCapture(0)

last_spoken = {}
cooldown = 5  # seconds

def speak(text):
    try:
        filename = "voice.mp3"

        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        playsound(filename)

        if os.path.exists(filename):
            os.remove(filename)

    except Exception as e:
        print("Voice Error:", e)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO Detection
    results = model(frame)

    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():

        label = row['name']
        conf = row['confidence']

        if conf > 0.5:

            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            current_time = time.time()

            if (label not in last_spoken or
                current_time - last_spoken[label] > cooldown):

                last_spoken[label] = current_time

                threading.Thread(
                    target=speak,
                    args=(f"{label} detected",),
                    daemon=True
                ).start()

    cv2.imshow("YOLOv5 Voice Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()