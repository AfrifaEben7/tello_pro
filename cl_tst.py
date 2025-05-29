#  _____       *  *        ___ **   **
# |_   *| *** | || | **_  / __|\ \ / /
#   | |  / -_)| || |/ * \| (*_  \   /
#   |_|  \___||_||_|\___/ \___|  \_/
# 
# Configuration flags:
# Set ENABLE_DISPLAY = False if you have OpenCV display issues
# Set TRACKING_MODE = True to follow the object while hovering
#
ENABLE_DISPLAY = True
TRACKING_MODE = True  # Follow object movements while hovering
TARGET_WIDTH = 250   # Pixels - how close to get (larger = closer)

# Import libraries
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
frozen_target = None  # Frozen target info after acquisition
object_lock = threading.Lock()
status_message = "Initializing..."
movement_status = ""
frame_display = None
display_active = True
detection_active = True
target_acquired = False

# Initializing the Tello drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

# State the path/location of where the video output file will be saved
video_path = 'test_img/vid_det.mp4'  

# State what video codec to use (mp4, h.264, av1, etc.)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_out = (video_path, fourcc, FPS, (Frame Width, Frame Height), isColor = T/F)
video_out = cv2.VideoWriter(video_path, fourcc, 20, (960,720), isColor=True)

# Load YOLO model
model = YOLO('best_yolov8n.pt', task='detect')

# Get the 'BackgroundFrameRead' object that HOLDS the latest captured frame from the Tello
frame_read = tello.get_frame_read(with_queue=False, max_queue_len=0)

def draw_status_info(frame, status, movement, target_obj):
    """Draw status information on the frame"""
    # Create a semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw status message
    cv2.putText(frame, f"Status: {status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw movement status
    cv2.putText(frame, f"Movement: {movement}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw target object info
    if target_obj:
        obj_info = f"Target: {target_obj['class']} (Conf: {target_obj['confidence']:.2f})"
        position = f"Position: ({int(target_obj['center'][0])}, {int(target_obj['center'][1])})"
        cv2.putText(frame, obj_info, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, position, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw target box
        x1, y1, x2, y2 = map(int, target_obj['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw crosshair on target
        cx, cy = map(int, target_obj['center'])
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        
        # Draw distance indicator
        obj_width = x2 - x1
        distance_text = f"Width: {obj_width}px (Target: {TARGET_WIDTH}px)"
        cv2.putText(frame, distance_text, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Visual distance bar
        bar_length = min(int(obj_width / TARGET_WIDTH * 200), 200)
        cv2.rectangle(frame, (450, 140), (450 + 200, 155), (100, 100, 100), 2)
        cv2.rectangle(frame, (450, 140), (450 + bar_length, 155), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No target detected", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw center crosshair
    cv2.drawMarker(frame, (320, 320), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    
    # Draw tracking mode indicator
    if TRACKING_MODE and target_acquired:
        cv2.putText(frame, "TRACKING ON", (520, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# Detection function that runs in a separate thread
def detection_thread():
    global highest_confidence_object, frame_display, frozen_target
    
    while detection_active:
        try:
            frame = frame_read.frame
            if frame is not None:
                # Resize frame
                frame_resized = cv2.resize(frame, (640, 640))
                # Convert from BGR to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Only run detection if target not acquired or in tracking mode
                if not target_acquired or (TRACKING_MODE and target_acquired):
                    # Run YOLOv8 Object Detection
                    results = model(frame_rgb, conf=0.70, verbose=False)
                    
                    # Process detection results
                    boxes = results[0].boxes
                    
                    # Find object with highest confidence
                    if len(boxes) > 0:
                        max_conf_idx = boxes.conf.argmax()
                        with object_lock:
                            highest_confidence_object = {
                                'class': results[0].names[int(boxes.cls[max_conf_idx])],
                                'confidence': float(boxes.conf[max_conf_idx]),
                                'bbox': boxes.xyxy[max_conf_idx].tolist(),
                                'center': calculate_center(boxes.xyxy[max_conf_idx].tolist())
                            }
                        if not target_acquired:
                            print(f"Detected: {highest_confidence_object['class']} ({highest_confidence_object['confidence']:.2f})")
                    else:
                        with object_lock:
                            if not target_acquired:
                                highest_confidence_object = None
                
                # Convert back to BGR for OpenCV display
                frame_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Use frozen target for display if target acquired
                display_target = frozen_target if target_acquired else highest_confidence_object
                
                # Add status information to frame
                with object_lock:
                    frame_display = draw_status_info(frame_display, status_message, movement_status, display_target)
                
                # Write frame to video
                video_out.write(frame_display)
                
                # Control frame rate
                time.sleep(1/10)
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.1)

def calculate_center(bbox):
    """Calculate center of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def approach_object(target_object):
    """Move drone to approach the target object until close"""
    global movement_status, status_message
    
    if target_object is None:
        status_message = "No object to approach"
        return False
    
    status_message = f"Approaching {target_object['class']}"
    
    # Get object center and frame center
    obj_x, obj_y = target_object['center']
    frame_center_x, frame_center_y = 320, 320  # Center of 640x640 frame
    
    # Calculate object size (for distance estimation)
    x1, y1, x2, y2 = target_object['bbox']
    obj_width = x2 - x1
    
    # Check if we're close enough
    if obj_width >= TARGET_WIDTH:
        movement_status = "Object reached - Close enough!"
        status_message = "At target distance"
        print(f"Object reached! Width: {int(obj_width)}px >= {TARGET_WIDTH}px")
        return True  
    
    # Centering the object horizontally
    horizontal_error = obj_x - frame_center_x
    if abs(horizontal_error) > 50:  # Dead zone of 50 pixels
        if horizontal_error > 0:
            movement_status = "Moving RIGHT to center object"
            print(f"Moving RIGHT - Error: {int(horizontal_error)}px")
            tello.move_right(30)
        else:
            movement_status = "Moving LEFT to center object"
            print(f"Moving LEFT - Error: {int(horizontal_error)}px")
            tello.move_left(30)
        return False
    
    # Centering the object vertically
    vertical_error = obj_y - frame_center_y
    if abs(vertical_error) > 50:  # Dead zone of 50 pixels
        if vertical_error > 0:
            movement_status = "Moving DOWN to center object"
            print(f"Moving DOWN - Error: {int(vertical_error)}px")
            tello.move_down(20)
        else:
            movement_status = "Moving UP to center object"
            print(f"Moving UP - Error: {int(vertical_error)}px")
            tello.move_up(20)
        return False
    
    # Move forward if object is small (far away)
    if obj_width < TARGET_WIDTH:
        movement_status = f"Moving FORWARD - Width: {int(obj_width)}px < {TARGET_WIDTH}px"
        print(f"Moving FORWARD - Width: {int(obj_width)}px < {TARGET_WIDTH}px")
        # Move forward more aggressively based on distance
        if obj_width < TARGET_WIDTH * 0.5:
            tello.move_forward(50)  # Far away - move 50cm
        elif obj_width < TARGET_WIDTH * 0.75:
            tello.move_forward(30)  # Medium distance - move 30cm
        else:
            tello.move_forward(20)  # Close - move 20cm
        return False
    
    return True  # We're close enough

def track_object(target_object):
    """Track object movements while hovering"""
    global movement_status
    
    if target_object is None:
        return
    
    # Get object center and frame center
    obj_x, obj_y = target_object['center']
    frame_center_x, frame_center_y = 320, 320
    
    # Check if object moved significantly
    horizontal_error = obj_x - frame_center_x
    vertical_error = obj_y - frame_center_y
    
    # Larger dead zone for tracking (100 pixels)
    if abs(horizontal_error) > 100:
        if horizontal_error > 0:
            movement_status = "Tracking: Moving RIGHT"
            print(f"Tracking: Moving RIGHT - Error: {int(horizontal_error)}px")
            tello.move_right(20)
        else:
            movement_status = "Tracking: Moving LEFT"
            print(f"Tracking: Moving LEFT - Error: {int(horizontal_error)}px")
            tello.move_left(20)
    elif abs(vertical_error) > 100:
        if vertical_error > 0:
            movement_status = "Tracking: Moving DOWN"
            print(f"Tracking: Moving DOWN - Error: {int(vertical_error)}px")
            tello.move_down(20)
        else:
            movement_status = "Tracking: Moving UP"
            print(f"Tracking: Moving UP - Error: {int(vertical_error)}px")
            tello.move_up(20)
    else:
        movement_status = "Tracking: Object centered"

# Main execution
try:
    # Start detection thread
    detection_thread_obj = threading.Thread(target=detection_thread)
    detection_thread_obj.start()
    
    # Create window in main thread if display is enabled
    if ENABLE_DISPLAY:
        try:
            cv2.namedWindow("YOLOv8 Tello Drone Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8 Tello Drone Tracking", 640, 640)
        except Exception as e:
            print(f"Could not create display window: {e}")
            print("Continuing without display...")
            ENABLE_DISPLAY = False
    
    # Helper function to update display
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
    
    # Take off
    status_message = "Taking off..."
    movement_status = "Ascending"
    print("Taking off...")
    tello.takeoff()
    
    # Wait for stable flight with display updates
    for _ in range(30):  # 3 seconds
        if not update_display():
            break
        time.sleep(0.1)
    
    # Perform rotation sequence while detecting
    status_message = "Starting rotation sequence"
    
    # Turn left
    status_message = "Scanning environment"
    movement_status = "Rotating LEFT 90°"
    print("Turning left...")
    tello.rotate_counter_clockwise(90)
    for _ in range(20):  # 2 seconds
        if not update_display():
            break
        time.sleep(0.1)
    
    # Turn right (back through center to right)
    movement_status = "Rotating RIGHT 180°"
    print("Turning right...")
    tello.rotate_clockwise(180)
    for _ in range(20):  # 2 seconds
        if not update_display():
            break
        time.sleep(0.1)
    
    # Turn back to center
    movement_status = "Returning to CENTER"
    print("Returning to center...")
    tello.rotate_counter_clockwise(90)
    for _ in range(20):  # 2 seconds
        if not update_display():
            break
        time.sleep(0.1)
    
    # Get the highest confidence object after rotations
    with object_lock:
        target = highest_confidence_object
        if target:
            frozen_target = target.copy()  # Freeze the target info
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
        
        # Wait a moment to show target info
        for _ in range(20):  # 2 seconds
            if not update_display():
                break
            time.sleep(0.1)
        
        # Approach the object until close
        close_enough = False
        max_attempts = 15  # More attempts for approach
        attempts = 0
        
        status_message = "Starting approach sequence"
        print("Starting approach to target...")
        
        while not close_enough and attempts < max_attempts and display_active:
            # Always use frozen target for approach
            close_enough = approach_object(frozen_target)
            attempts += 1
            
            # Wait between movements
            for _ in range(10):  # 1 second
                if not update_display():
                    display_active = False
                    break
                time.sleep(0.1)
        
        if close_enough:
            status_message = "Target reached - Hovering"
            movement_status = "Maintaining position"
            print("\nSuccessfully reached target!")
            
            # Hover and track if enabled
            if TRACKING_MODE:
                print("Tracking mode active - following object movements")
                status_message = "Tracking mode active"
                
                # Track for 10 seconds
                for i in range(100):  # 10 seconds
                    if not update_display():
                        break
                    
                    # Update tracking every 10 frames (1 second)
                    if i % 10 == 0:
                        with object_lock:
                            current_target = highest_confidence_object
                        if current_target and current_target['class'] == target['class']:
                            track_object(current_target)
                    
                    time.sleep(0.1)
            else:
                # Just hover without tracking
                print("Hovering at target (tracking disabled)")
                for _ in range(50):  # 5 seconds
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
        
        # Hover for a moment
        for _ in range(30):  # 3 seconds
            if not update_display():
                break
            time.sleep(0.1)
    
    # Land
    status_message = "Landing..."
    movement_status = "Descending"
    print("\nLanding...")
    tello.land()
    
    status_message = "Mission complete"
    movement_status = "Landed"
    for _ in range(20):  # 2 seconds
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
    # Stop threads
    detection_active = False
    display_active = False
    
    # Wait for threads to finish
    if 'detection_thread_obj' in locals():
        detection_thread_obj.join(timeout=2)
    
    # Clean up
    video_out.release()
    tello.end()
    cv2.destroyAllWindows()
    print("Cleanup complete")

# Configuration options:
# - ENABLE_DISPLAY: Set to False if you have OpenCV display issues
# - TRACKING_MODE: Set to True to follow object movements while hovering
# - TARGET_WIDTH: Adjust to control how close the drone gets (larger = closer)
#
# The drone will:
# 1. Take off and scan the environment
# 2. Acquire the highest confidence object
# 3. Freeze the target info and approach it
# 4. Stop when the object width reaches TARGET_WIDTH pixels
# 5. If TRACKING_MODE is True, follow the object if it moves
# 6. Land after hovering