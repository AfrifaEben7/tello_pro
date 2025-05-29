import cv2
from ultralytics import YOLO
from djitellopy import Tello
import threading
import time
import numpy as np
import os

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

# === CONFIGURATION ===
TARGET_CLASS = "chair"   
MODEL_PATH = "yolov8n.pt"  
DETECTION_SIZE = 320
DETECTION_INTERVAL = 2   # Detect every 2nd frame for speed
TARGET_WIDTH = 140       # How close is "close enough" (pixels at 320x320)
MAX_APPROACH_ATTEMPTS = 15


tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Battery: {battery}%")
if battery < 20:
    print("Battery too low for flight.")
    tello.end()
    exit()
tello.streamon()
frame_read = tello.get_frame_read()


model = YOLO(MODEL_PATH)


detection_active = True
def keep_alive_thread():
    while detection_active:
        try:
            tello.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"Keep-alive error: {e}")
        time.sleep(0.6)
ka_thread = threading.Thread(target=keep_alive_thread, daemon=True)
ka_thread.start()

# === MAIN LOGIC ===
def approach_object(obj_center, box_width):
    cx, cy = obj_center
    frame_cx, frame_cy = DETECTION_SIZE//2, DETECTION_SIZE//2
    moved = False
    # Center horizontally
    err_x = cx - frame_cx
    if abs(err_x) > 20:
        if err_x > 0:
            print("Moving RIGHT to center")
            tello.move_right(20)
        else:
            print("Moving LEFT to center")
            tello.move_left(20)
        moved = True
    # Center vertically
    err_y = cy - frame_cy
    if abs(err_y) > 20:
        if err_y > 0:
            print("Moving DOWN to center")
            tello.move_down(20)
        else:
            print("Moving UP to center")
            tello.move_up(20)
        moved = True
    # Approach if not close enough
    if not moved and box_width < TARGET_WIDTH:
        print(f"Moving FORWARD (box width {box_width} < {TARGET_WIDTH})")
        tello.move_forward(30)
        moved = True
    return box_width >= TARGET_WIDTH

try:
    print("Taking off...")
    tello.takeoff()
    time.sleep(2)
    frame_count = 0
    close_enough = False
    approach_attempts = 0

    while detection_active and not close_enough and approach_attempts < MAX_APPROACH_ATTEMPTS:
        frame = frame_read.frame
        if frame is None:
            continue
        small = cv2.resize(frame, (DETECTION_SIZE, DETECTION_SIZE))
        frame_disp = small.copy()
        found = False
        if frame_count % DETECTION_INTERVAL == 0:
            results = model(small)
            boxes = results[0].boxes
            names = results[0].names
            best = None
            best_conf = 0
            for i in range(len(boxes.cls)):
                cls_name = names[int(boxes.cls[i])]
                if cls_name == TARGET_CLASS and boxes.conf[i] > best_conf:
                    best = boxes.xyxy[i].tolist()
                    best_conf = boxes.conf[i]
            if best:
                x1, y1, x2, y2 = map(int, best)
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                width = x2 - x1
                found = True
                # Draw box and crosshair
                cv2.rectangle(frame_disp, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.drawMarker(frame_disp, (cx,cy), (255,0,0), cv2.MARKER_CROSS, 20, 2)
                print(f"Detected {TARGET_CLASS.upper()} (conf {best_conf:.2f}), width={width}")
                close_enough = approach_object((cx, cy), width)
                approach_attempts += 1
            else:
                print(f"No {TARGET_CLASS} detected.")
        # Optional: show frame
        cv2.imshow("Detection", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
        frame_count += 1
        time.sleep(0.2)

    print("Landing...")
    tello.land()
    time.sleep(2)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    detection_active = False
    tello.end()
    cv2.destroyAllWindows()
    print("Done.")

