import cv2
from ultralytics import YOLO
from djitellopy import Tello
import os
import time
import threading
import numpy as np

# ==== CONFIGURATION ====
MODEL_PATH = 'best_yolov8n.pt'
FRAME_SIZE = (640, 640)
FRAME_CENTER = (FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2)
TARGET_WIDTH = 300  # px - how close to get before stopping approach
CONF_THRESHOLD = 0.70
STICKY_FRAMES = 5   # Num. frames to retain last target if detection drops
ROTATION_STEP = 30  # deg for each scan rotation
MAX_SCAN_ATTEMPTS = 12
ENABLE_TRACKING = True
MAX_APPROACH_ATTEMPTS = 15
DEADZONE_PX = 50
SAFETY_MAX_LOST = 10  # Max frames allowed to lose feed or target
ENABLE_DISPLAY = True
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

# ==== GLOBAL STATE ====
state_lock = threading.Lock()
target = None
last_seen_target = None
frames_since_seen = 0
status_message = "Initializing..."
movement_status = ""
frame_display = None
detection_active = True
display_active = True

# ==== INIT DRONE AND YOLO ====
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()
model = YOLO(MODEL_PATH, task='detect')
frame_read = tello.get_frame_read(with_queue=False, max_queue_len=0)

# ==== UTILITY FUNCTIONS ====

def draw_status_info(frame, status, movement, tgt):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_SIZE[0], 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Movement: {movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if tgt:
        obj_info = f"Target: {tgt['class']} (Conf: {tgt['confidence']:.2f})"
        position = f"Position: ({int(tgt['center'][0])}, {int(tgt['center'][1])})"
        cv2.putText(frame, obj_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, position, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        x1, y1, x2, y2 = map(int, tgt['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cx, cy = map(int, tgt['center'])
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        obj_width = x2 - x1
        bar_len = min(int(obj_width / TARGET_WIDTH * 200), 200)
        cv2.rectangle(frame, (420, 120), (620, 135), (100, 100, 100), 2)
        cv2.rectangle(frame, (420, 120), (420 + bar_len, 135), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No target detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.drawMarker(frame, FRAME_CENTER, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    return frame

def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def detect_highest_confidence_object(frame_rgb):
    results = model(frame_rgb, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        max_idx = boxes.conf.argmax()
        bbox = boxes.xyxy[max_idx].tolist()
        center = calculate_center(bbox)
        return {
            'class': results[0].names[int(boxes.cls[max_idx])],
            'confidence': float(boxes.conf[max_idx]),
            'bbox': bbox,
            'center': center
        }
    return None

# ==== THREADS ====

def detection_thread():
    global last_seen_target, frames_since_seen, frame_display
    while detection_active:
        try:
            frame = frame_read.frame
            if frame is not None:
                frame_resized = cv2.resize(frame, FRAME_SIZE)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                current = detect_highest_confidence_object(frame_rgb)
                with state_lock:
                    if current:
                        last_seen_target = current
                        frames_since_seen = 0
                    else:
                        frames_since_seen += 1
                frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with state_lock:
                    display_target = last_seen_target if frames_since_seen <= STICKY_FRAMES else None
                    frame_display = draw_status_info(frame_display, status_message, movement_status, display_target)
            time.sleep(1/10)
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)

# ==== STATE MACHINE LOGIC ====

def scan_and_acquire():
    global status_message, movement_status
    for i in range(MAX_SCAN_ATTEMPTS):
        status_message = "Scanning..."
        movement_status = f"Rotating {ROTATION_STEP} deg"
        tello.rotate_counter_clockwise(ROTATION_STEP)
        for _ in range(7):  # Pause for camera/detection update
            time.sleep(0.1)
        with state_lock:
            if last_seen_target and last_seen_target['confidence'] >= CONF_THRESHOLD:
                return last_seen_target
    return None

def approach_object():
    global movement_status, status_message
    approach_attempts = 0
    while approach_attempts < MAX_APPROACH_ATTEMPTS:
        with state_lock:
            tgt = last_seen_target if frames_since_seen <= STICKY_FRAMES else None
        if not tgt:
            status_message = "Target lost - Stopping approach"
            movement_status = "Hovering"
            return False
        obj_x, obj_y = tgt['center']
        x1, y1, x2, y2 = tgt['bbox']
        obj_width = x2 - x1
        if obj_width >= TARGET_WIDTH:
            status_message = "At target distance"
            movement_status = "Target reached"
            return True
        # Center horizontally
        horiz_error = obj_x - FRAME_CENTER[0]
        vert_error = obj_y - FRAME_CENTER[1]
        if abs(horiz_error) > DEADZONE_PX:
            if horiz_error > 0:
                movement_status = "Right"
                tello.move_right(30)
            else:
                movement_status = "Left"
                tello.move_left(30)
        elif abs(vert_error) > DEADZONE_PX:
            if vert_error > 0:
                movement_status = "Down"
                tello.move_down(20)
            else:
                movement_status = "Up"
                tello.move_up(20)
        else:
            movement_status = "Forward"
            tello.move_forward(30)
        approach_attempts += 1
        time.sleep(1.2)
    return False

def tracking_loop(target_class):
    global movement_status
    print("Tracking mode enabled.")
    for _ in range(100):  # Track for ~10 sec
        with state_lock:
            tgt = last_seen_target if frames_since_seen <= STICKY_FRAMES else None
        if not tgt or tgt['class'] != target_class:
            movement_status = "Tracking lost"
            break
        obj_x, obj_y = tgt['center']
        horiz_error = obj_x - FRAME_CENTER[0]
        vert_error = obj_y - FRAME_CENTER[1]
        if abs(horiz_error) > DEADZONE_PX*2:
            if horiz_error > 0:
                movement_status = "Track: Right"
                tello.move_right(20)
            else:
                movement_status = "Track: Left"
                tello.move_left(20)
        elif abs(vert_error) > DEADZONE_PX*2:
            if vert_error > 0:
                movement_status = "Track: Down"
                tello.move_down(20)
            else:
                movement_status = "Track: Up"
                tello.move_up(20)
        else:
            movement_status = "Track: Centered"
        time.sleep(0.7)

def update_display():
    global ENABLE_DISPLAY
    if ENABLE_DISPLAY and frame_display is not None:
        try:
            cv2.imshow("YOLOv8 Tello Drone Tracking", frame_display)
            if cv2.waitKey(1) & 0xFF == ord("x"):
                return False
        except Exception as e:
            print(f"Display error: {e}")
            ENABLE_DISPLAY = False
    return True



try:
    detection_thread_obj = threading.Thread(target=detection_thread, daemon=True)
    detection_thread_obj.start()
    if ENABLE_DISPLAY:
        try:
            cv2.namedWindow("YOLOv8 Tello Drone Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Tello Drone Tracking", FRAME_SIZE[0], FRAME_SIZE[1])
        except Exception as e:
            print(f"Could not create display window: {e}")
            ENABLE_DISPLAY = False
    print("Taking off...")
    status_message = "Taking off"
    tello.takeoff()
    for _ in range(30):
        if not update_display():
            break
        time.sleep(0.1)

    # SCAN & ACQUIRE
    status_message = "Scanning for target"
    acquired = scan_and_acquire()
    if acquired:
        print(f"Target acquired: {acquired['class']} ({acquired['confidence']:.2f})")
        status_message = f"Target acquired: {acquired['class']}"
    else:
        print("No target found. Hovering.")
        status_message = "No target found"
        tello.land()
        raise SystemExit

    # APPROACH
    print("Approaching target...")
    status_message = "Approaching target"
    approached = approach_object()
    if approached:
        print("Target reached!")
        status_message = "Target reached"
    else:
        print("Failed to approach. Hovering.")
        status_message = "Failed to approach"

    # TRACKING
    if ENABLE_TRACKING and approached:
        tracking_loop(acquired['class'])

    print("Landing...")
    status_message = "Landing"
    tello.land()
    for _ in range(20):
        if not update_display():
            break
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user")
    status_message = "User interrupted"
except Exception as e:
    print(f"Error occurred: {e}")
    status_message = f"Error: {str(e)}"
finally:
    detection_active = False
    display_active = False
    if 'detection_thread_obj' in locals():
        detection_thread_obj.join(timeout=2)
    tello.end()
    cv2.destroyAllWindows()
    print("Cleanup complete")
