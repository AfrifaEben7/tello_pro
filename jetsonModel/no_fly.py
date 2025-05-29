import cv2
from ultralytics import YOLO
from djitellopy import Tello
import time
import numpy as np

MODEL_PATH = "" 
TARGET_CLASSES = ["chair"] 

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
frame_read = tello.get_frame_read()

model = YOLO(MODEL_PATH)

try:
    print("Starting video stream and detection... (press x to exit)")
    while True:
        frame = frame_read.frame
        if frame is None:
            print("No frame received")
            time.sleep(0.1)
            continue

        small = cv2.resize(frame, (320, 320))
        results = model(small)
        boxes = results[0].boxes
        names = results[0].names

        for i in range(len(boxes.cls)):
            class_name = names[int(boxes.cls[i])]
            conf = boxes.conf[i]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            if class_name in TARGET_CLASSES or not TARGET_CLASSES:
                cv2.rectangle(small, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(small, f"{class_name} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                print(f"Detected {class_name} ({conf:.2f}) at {x1},{y1},{x2},{y2}")

        cv2.imshow("Tello Video + YOLO Detection", small)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

        time.sleep(0.01)

except KeyboardInterrupt:
    print("User interrupted.")
finally:
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
    print("Cleanup complete.")
