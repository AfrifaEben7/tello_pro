from djitellopy import Tello
import cv2
import time
import argparse
import os
from datetime import datetime

print("Create Tello object")
tello = Tello()

print("Connect to Tello Drone")
tello.connect()

battery_level = tello.get_battery()
print(f"Battery Life Percentage: {battery_level}")

time.sleep(2)

print("Turn Video Stream On")
tello.streamon()

# read a single image from the Tello video feed
print("Read Tello Image")
frame_read = tello.get_frame_read()

time.sleep(5) 

# Argument parsing
parser = argparse.ArgumentParser(description='Tello video stream saver')
parser.add_argument('--im', action='store_true', help='Save frames as images')
parser.add_argument('--vid', action='store_true', help='Save stream as video')
args = parser.parse_args()

# Create output directory for this run
run_dir = os.path.join('img', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(run_dir, exist_ok=True)

video_writer = None
frame_count = 0

while True:
    # read a single image from the Tello video feed
    print("Read Tello Image")
    tello_video_image = frame_read.frame

    if tello_video_image is not None:
        cv2.imshow("TelloVideo", tello_video_image)
        
        # images
        if args.im:
            filename = os.path.join(run_dir, f"img_{frame_count:04d}.jpg")
            cv2.imwrite(filename, tello_video_image)
            
        # vid
        if args.vid:
            if video_writer is None:
                height, width, _ = tello_video_image.shape
                video_writer = cv2.VideoWriter(
                    os.path.join(run_dir, 'output.avi'),
                    cv2.VideoWriter_fourcc(*'XVID'),
                    10,  # FPS
                    (width, height)
                )
            video_writer.write(tello_video_image)
        if args.im or args.vid:
            frame_count += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    time.sleep(1/5)

if video_writer is not None:
    video_writer.release()

tello.streamoff()
cv2.destroyWindow('TelloVideo')
cv2.destroyAllWindows()