from ultralytics import YOLO

yolo = YOLO('/Users/eben/Desktop/sdsmt/Projects/Tello/best_yolov8n.pt', task='detect')


yolo.to('mps')  # to use cuda, use yolo.to('cuda')

directory = '/Users/eben/Desktop/sdsmt/Projects/Tello/test_img'

# running inference on images in the directory
results = yolo.predict(source=directory, conf=0.5, show=True, save=True, project=directory)
