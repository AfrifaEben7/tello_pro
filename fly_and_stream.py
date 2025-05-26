from djitellopy import Tello
import cv2
import time
import os
from datetime import datetime

# Create Tello object and connect
print("Create Tello object")
tello = Tello()
print("Connect to Tello Drone")
tello.connect()

battery_level = tello.get_battery()
print(f"Battery Life Percentage: {battery_level}")

# Start video stream
print("Turn Video Stream On")
tello.streamon()
frame_read = tello.get_frame_read()

# Create output directory for this run
run_dir = os.path.join('img', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(run_dir, exist_ok=True)
video_path = os.path.join(run_dir, 'output.avi')
video_writer = None

# Takeoff
print("Takeoff")
tello.takeoff()
time.sleep(2)


print("Fly up")
tello.move_up(50)  # Move up 50cm

# Stream video for 10 seconds while flying up and down
start_time = time.time()
while time.time() - start_time < 10:
    frame = frame_read.frame
    if frame is not None:
        cv2.imshow("TelloVideo", frame)
        if video_writer is None:
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(
                os.path.join(run_dir, 'output.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,  # FPS
                (width, height)
            )
        video_writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1/5)  # 5 FPS

# Fly down
print("Fly down")
tello.move_down(50) 



# Cleanup
print("Turn Video Stream Off")
tello.streamoff()
cv2.destroyWindow('TelloVideo')
cv2.destroyAllWindows()
if video_writer is not None:
    video_writer.release()
