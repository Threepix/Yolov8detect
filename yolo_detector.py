import torch
from ultralytics import YOLO

class YOLODetector:
    """
    YOLODetector uses YOLOv8 model for detecting objects in frames.
    """
    def __init__(self, model_name='yolov8s'):
        self.model = YOLO(model_name)
    
    def detect(self, frame):
        results = self.model(frame)
        return results.pandas().xyxy[0]
