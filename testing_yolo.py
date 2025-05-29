from ultralytics import YOLO

yolo = YOLO('best_yolov8n.pt', task='detect')


yolo.to('mps') 

directory = 'test_img'

# running inference on images in the directory
results = yolo.predict(source=directory, conf=0.5, show=True, save=True, project=directory)
