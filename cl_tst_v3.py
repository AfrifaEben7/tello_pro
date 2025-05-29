#  _____       *  *        ___ **   **
# |_   *| *** | || | **_  / __|\ \ / /
#   | |  / -_)| || |/ * \| (*_  \   /
#   |_|  \___||_||_|\___/ \___|  \_/
#
# Tello Drone Object Detection and Tracking with Sticky Target & Safety

ENABLE_DISPLAY = True
TRACKING_MODE = True
TURN_ENABLED = False
TARGET_WIDTH = 250

LOST_TOLERANCE = 3  # Number of consecutive lost frames before aborting approach
MIN_BATTERY = 20    # Minimum battery percent to start/continue mission

import cv2
from ultralytics import YOLO
from djitellopy import Tello
import os
import time
import threading
import numpy as np

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

# Global variables for tracking
highest_confidence_object = None
frozen_target = None
object_lock = threading.Lock()
status_message = "Initializing..."
movement_status = ""
frame_display = None
display_active = True
detection_active = True
target_acquired = False
lost_counter = 0

# Initializing the Tello drone
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Battery: {battery}%")
if battery < MIN_BATTERY:
    print("Battery too low for mission. Exiting.")
    tello.end()
    exit()
tello.streamon()

video_path = 'test_img/vid_det.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(video_path, fourcc, 20, (640, 640), isColor=True)

model = YOLO('yolov8n.pt', task='detect')
frame_read = tello.get_frame_read(with_queue=False, max_queue_len=0)

def draw_status_info(frame, status, movement, target_obj):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Movement: {movement}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if target_obj:
        obj_info = f"Target: {target_obj['class']} (Conf: {target_obj['confidence']:.2f})"
        position = f"Position: ({int(target_obj['center'][0])}, {int(target_obj['center'][1])})"
        cv2.putText(frame, obj_info, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, position, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        x1, y1, x2, y2 = map(int, target_obj['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cx, cy = map(int, target_obj['center'])
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        obj_width = x2 - x1
        distance_text = f"Width: {obj_width}px (Target: {TARGET_WIDTH}px)"
        cv2.putText(frame, distance_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        bar_length = min(int(obj_width / TARGET_WIDTH * 200), 200)
        cv2.rectangle(frame, (450, 140), (450 + 200, 155), (100, 100, 100), 2)
        cv2.rectangle(frame, (450, 140), (450 + bar_length, 155), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No target detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.drawMarker(frame, (320, 320), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    if TRACKING_MODE and target_acquired:
        cv2.putText(frame, "TRACKING ON", (520, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def detection_thread():
    global highest_confidence_object, frame_display, frozen_target, lost_counter
    while detection_active:
        try:
            frame = frame_read.frame
            if frame is not None:
                frame_resized = cv2.resize(frame, (640, 640))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                if not target_acquired or (TRACKING_MODE and target_acquired):
                    results = model(frame_rgb, conf=0.70, verbose=False)
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        max_conf_idx = boxes.conf.argmax()
                        with object_lock:
                            highest_confidence_object = {
                                'class': results[0].names[int(boxes.cls[max_conf_idx])],
                                'confidence': float(boxes.conf[max_conf_idx]),
                                'bbox': boxes.xyxy[max_conf_idx].tolist(),
                                'center': calculate_center(boxes.xyxy[max_conf_idx].tolist())
                            }
                        lost_counter = 0
                        if not target_acquired:
                            print(f"Detected: {highest_confidence_object['class']} ({highest_confidence_object['confidence']:.2f})")
                    else:
                        with object_lock:
                            if not target_acquired:
                                highest_confidence_object = None
                        lost_counter += 1
                frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                display_target = frozen_target if target_acquired else highest_confidence_object
                with object_lock:
                    frame_display = draw_status_info(frame_display, status_message, movement_status, display_target)
                video_out.write(frame_display)
                time.sleep(1/10)
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)

def calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_sticky_target():
    global highest_confidence_object, frozen_target, lost_counter
    with object_lock:
        if highest_confidence_object:
            frozen_target = highest_confidence_object.copy()
            lost_counter = 0
        else:
            lost_counter += 1
        if lost_counter <= LOST_TOLERANCE:
            return frozen_target
        return None

def safe_move(fn, *args):
    try:
        fn(*args)
    except Exception as move_err:
        print(f"Drone move error: {move_err}")

def approach_object(target_object):
    global movement_status, status_message
    if target_object is None:
        status_message = "No object to approach"
        return False
    status_message = f"Approaching {target_object['class']}"
    obj_x, obj_y = target_object['center']
    frame_center_x, frame_center_y = 320, 320
    x1, y1, x2, y2 = target_object['bbox']
    obj_width = x2 - x1
    if obj_width >= TARGET_WIDTH:
        movement_status = "Object reached - Close enough!"
        status_message = "At target distance"
        print(f"Object reached! Width: {int(obj_width)}px >= {TARGET_WIDTH}px")
        return True
    horizontal_error = obj_x - frame_center_x
    if abs(horizontal_error) > 50:
        if horizontal_error > 0:
            movement_status = "Moving RIGHT to center object"
            print(f"Moving RIGHT - Error: {int(horizontal_error)}px")
            safe_move(tello.move_right, 30)
        else:
            movement_status = "Moving LEFT to center object"
            print(f"Moving LEFT - Error: {int(horizontal_error)}px")
            safe_move(tello.move_left, 30)
        return False
    vertical_error = obj_y - frame_center_y
    if abs(vertical_error) > 50:
        if vertical_error > 0:
            movement_status = "Moving DOWN to center object"
            print(f"Moving DOWN - Error: {int(vertical_error)}px")
            safe_move(tello.move_down, 20)
        else:
            movement_status = "Moving UP to center object"
            print(f"Moving UP - Error: {int(vertical_error)}px")
            safe_move(tello.move_up, 20)
        return False
    if obj_width < TARGET_WIDTH:
        movement_status = f"Moving FORWARD - Width: {int(obj_width)}px < {TARGET_WIDTH}px"
        print(f"Moving FORWARD - Width: {int(obj_width)}px < {TARGET_WIDTH}px")
        if obj_width < TARGET_WIDTH * 0.5:
            safe_move(tello.move_forward, 50)
        elif obj_width < TARGET_WIDTH * 0.75:
            safe_move(tello.move_forward, 30)
        else:
            safe_move(tello.move_forward, 20)
        return False
    return True

def track_object(target_object):
    global movement_status
    if target_object is None:
        return
    obj_x, obj_y = target_object['center']
    frame_center_x, frame_center_y = 320, 320
    horizontal_error = obj_x - frame_center_x
    vertical_error = obj_y - frame_center_y
    if abs(horizontal_error) > 100:
        if horizontal_error > 0:
            movement_status = "Tracking: Moving RIGHT"
            print(f"Tracking: Moving RIGHT - Error: {int(horizontal_error)}px")
            safe_move(tello.move_right, 20)
        else:
            movement_status = "Tracking: Moving LEFT"
            print(f"Tracking: Moving LEFT - Error: {int(horizontal_error)}px")
            safe_move(tello.move_left, 20)
    elif abs(vertical_error) > 100:
        if vertical_error > 0:
            movement_status = "Tracking: Moving DOWN"
            print(f"Tracking: Moving DOWN - Error: {int(vertical_error)}px")
            safe_move(tello.move_down, 20)
        else:
            movement_status = "Tracking: Moving UP"
            print(f"Tracking: Moving UP - Error: {int(vertical_error)}px")
            safe_move(tello.move_up, 20)
    else:
        movement_status = "Tracking: Object centered"

try:
    detection_thread_obj = threading.Thread(target=detection_thread)
    detection_thread_obj.start()

    print("\n" + "="*50)
    print("DRONE CONFIGURATION:")
    print(f"- Turn sequence: {'ENABLED' if TURN_ENABLED else 'DISABLED'}")
    print(f"- Tracking mode: {'ENABLED' if TRACKING_MODE else 'DISABLED'}")
    print(f"- Target width: {TARGET_WIDTH}px")
    print(f"- Display: {'ENABLED' if ENABLE_DISPLAY else 'DISABLED'}")
    print("="*50 + "\n")

    if ENABLE_DISPLAY:
        try:
            cv2.namedWindow("YOLOv8 Tello Drone Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Tello Drone Tracking", 640, 640)
        except Exception as e:
            print(f"Could not create display window: {e}")
            ENABLE_DISPLAY = False

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

    status_message = "Taking off..."
    movement_status = "Ascending"
    print("Taking off...")
    tello.takeoff()
    for _ in range(30):
        if not update_display():
            break
        time.sleep(0.1)

    # Skipping rotation, just wait a moment for detection to start
    status_message = "Detecting objects ahead"
    movement_status = "Scanning front view"
    print("Skipping rotation - detecting objects in front...")
    for _ in range(30):
        if not update_display():
            break
        time.sleep(0.1)

    # Get the highest confidence object after rotations
    with object_lock:
        target = highest_confidence_object
        if target:
            frozen_target = target.copy()
            target_acquired = True

    if target:
        status_message = f"Target acquired: {target['class']}"
        print(f"\n{'='*50}")
        print(f"TARGET ACQUIRED: {target['class']}")
        print(f"Confidence: {target['confidence']:.2f}")
        print(f"Position: ({int(target['center'][0])}, {int(target['center'][1])})")
        print(f"Width: {int(target['bbox'][2] - target['bbox'][0])}px")
        print(f"Target width for approach: {TARGET_WIDTH}px")
        print(f"{'='*50}\n")
        for _ in range(20):
            if not update_display():
                break
            time.sleep(0.1)

        close_enough = False
        max_attempts = 15
        attempts = 0

        status_message = "Starting approach sequence"
        print("Starting approach to target...")

        while not close_enough and attempts < max_attempts and display_active:
            current_target = get_sticky_target()
            if current_target is None:
                print("Lost target - aborting approach and landing.")
                status_message = "Lost target, landing"
                tello.land()
                break
            close_enough = approach_object(current_target)
            attempts += 1
            for _ in range(10):
                if not update_display():
                    display_active = False
                    break
                time.sleep(0.1)

        if close_enough:
            status_message = "Target reached - Hovering"
            movement_status = "Maintaining position"
            print("\nSuccessfully reached target!")
            if TRACKING_MODE:
                print("Tracking mode active - following object movements")
                status_message = "Tracking mode active"
                for i in range(100):
                    if not update_display():
                        break
                    if i % 10 == 0:
                        with object_lock:
                            current_target = highest_confidence_object
                        if current_target and current_target['class'] == target['class']:
                            track_object(current_target)
                    time.sleep(0.1)
            else:
                print("Hovering at target (tracking disabled)")
                for _ in range(50):
                    if not update_display():
                        break
                    time.sleep(0.1)
        else:
            status_message = "Could not reach target"
            movement_status = "Approach incomplete"
            print("Could not reach target within attempt limit")
    else:
        status_message = "No objects detected to approach"
        movement_status = "No movement needed"
        print("No objects detected to approach")
        for _ in range(30):
            if not update_display():
                break
            time.sleep(0.1)

    status_message = "Landing..."
    movement_status = "Descending"
    print("\nLanding...")
    tello.land()
    status_message = "Mission complete"
    movement_status = "Landed"
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
    video_out.release()
    tello.end()
    if ENABLE_DISPLAY:
        cv2.destroyAllWindows()
    print("Cleanup complete")
