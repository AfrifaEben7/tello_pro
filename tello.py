import cv2
from ultralytics import YOLO
from djitellopy import Tello
import os
import time
import numpy as np

os.environ["OPENCV_FFMPEG_DEBUG"] = "0"

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()

model = YOLO('/Users/eben/Desktop/sdsmt/Projects/Tello/best_yolov8n.pt', task='detect')

frame_read = tello.get_frame_read(with_queue=False, max_queue_len=0)

FRAME_WIDTH = 640
FRAME_HEIGHT = 640
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

def draw_detection(frame, obj):
    x1, y1, x2, y2 = obj["box"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
    label = f"{obj['class_name']} {obj['confidence']:.2f}"
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

def put_text_lines(frame, lines, start_y=30, line_height=30):
    """Helper to put multiple lines of text on frame"""
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, start_y + i*line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

def detect_objects(frame, conf_thresh=0.7):
    results = model(frame, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes
    detected_objects = []
    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = results[0].names[int(cls_id)]
        detected_objects.append({
            "class_name": class_name,
            "confidence": float(conf),
            "box": (x1, y1, x2, y2)
        })
    return detected_objects

def center_object_control(obj):
    x1, y1, x2, y2 = obj["box"]
    obj_center_x = (x1 + x2) // 2
    error_x = obj_center_x - CENTER_X
    dead_zone = 20  # pixels
    if abs(error_x) < dead_zone:
        return 0
    rotation_deg = int(np.clip(error_x * 0.1, -10, 10))
    return rotation_deg

def yaw_rotate(tello, degrees, clockwise=True, yaw_speed=40):
    """
    Rotate the drone by 'degrees' using yaw velocity commands.
    yaw_speed: degrees per second (positive int)
    """
    duration = abs(degrees) / yaw_speed  # seconds to rotate desired degrees
    if clockwise:
        yaw_vel = yaw_speed
    else:
        yaw_vel = -yaw_speed

    # Start rotation
    tello.send_rc_control(0, 0, 0, yaw_vel)
    time.sleep(duration)
    # Stop rotation
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)  # short pause for stability

def perform_rotations_with_detection(total_degrees, clockwise=True, step=10, conf_thresh=0.7):
    steps = total_degrees // step
    best_obj = None
    best_conf = 0

    for i in range(steps):
        action = f"Rotating {'right' if clockwise else 'left'} {step}°"
        yaw_rotate(tello, step, clockwise=clockwise)

        frame = frame_read.frame
        if frame is None:
            print("No frame during rotation step")
            continue
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        detected_objects = detect_objects(frame_rgb, conf_thresh)

        info_lines = [action]

        if detected_objects:
            step_best = max(detected_objects, key=lambda x: x["confidence"])
            if step_best["confidence"] > best_conf:
                best_conf = step_best["confidence"]
                best_obj = step_best

            draw_detection(frame_resized, step_best)
            obj_center = ((step_best["box"][0] + step_best["box"][2])//2,
                          (step_best["box"][1] + step_best["box"][3])//2)
            info_lines.append(f"Best obj: {step_best['class_name']} ({step_best['confidence']:.2f})")
            info_lines.append(f"Pos: {obj_center}")

        else:
            info_lines.append("No objects detected")

        put_text_lines(frame_resized, info_lines)
        cv2.imshow("Rotating Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    return best_obj

def approach_object(obj, conf_thresh=0.7):
    CLOSE_ENOUGH_AREA = 20000

    while True:
        frame = frame_read.frame
        if frame is None:
            print("Lost video feed during approach")
            break
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        detected_objects = detect_objects(frame_rgb, conf_thresh)
        if not detected_objects:
            print("Lost object during approach")
            break

        obj = max(detected_objects, key=lambda x: x["confidence"])

        x1, y1, x2, y2 = obj["box"]
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        draw_detection(frame_resized, obj)

        info_lines = [
            f"Best obj: {obj['class_name']} ({obj['confidence']:.2f})",
            f"Object area: {box_area}"
        ]

        if box_area >= CLOSE_ENOUGH_AREA:
            info_lines.append("Close enough to the object. Stopping.")
            put_text_lines(frame_resized, info_lines)
            cv2.imshow("Approach Object", frame_resized)
            cv2.waitKey(3000)  # pause 3 sec to read message
            break

        rotation_deg = center_object_control(obj)
        if rotation_deg != 0:
            if rotation_deg > 0:
                yaw_rotate(tello, abs(rotation_deg), clockwise=True)
                info_lines.append(f"Rotating right {abs(rotation_deg)}° to center")
            else:
                yaw_rotate(tello, abs(rotation_deg), clockwise=False)
                info_lines.append(f"Rotating left {abs(rotation_deg)}° to center")
        else:
            info_lines.append("Object centered")

        tello.move_forward(20)
        info_lines.append("Moving forward 20 cm")
        time.sleep(2)

        put_text_lines(frame_resized, info_lines)
        cv2.imshow("Approach Object", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

def main():
    print("Taking off...")
    tello.takeoff()
    time.sleep(3)

    obj_left = perform_rotations_with_detection(90, clockwise=False, step=10)
    obj_right = perform_rotations_with_detection(180, clockwise=True, step=10)
    obj_front = perform_rotations_with_detection(90, clockwise=False, step=10)

    all_objs = [obj for obj in [obj_left, obj_right, obj_front] if obj is not None]

    if not all_objs:
        print("No objects detected during rotations.")
    else:
        best_obj = max(all_objs, key=lambda x: x["confidence"])
        print(f"Best object detected: {best_obj['class_name']} ({best_obj['confidence']:.2f})")
        approach_object(best_obj)

    print("Landing...")
    tello.land()
    tello.end()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
