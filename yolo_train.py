from ultralytics import YOLO
import multiprocessing
# Load a model

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    multiprocessing.freeze_support()
    results = model.train(data="./coco128.yaml", epochs=10, batch=12 ,imgsz=640)
