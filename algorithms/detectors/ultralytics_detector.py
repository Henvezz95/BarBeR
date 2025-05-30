from ultralytics import YOLO, RTDETR
from time import perf_counter_ns
from detectors_abs import BaseDetector
import os


class YOLO_detector(BaseDetector):
    def __init__(self, model_path, imgsz, device = 'cpu'):
        if model_path.split('.')[-1] == 'onnx':
            self.model = YOLO(model_path, 
                              task='detect')
        else:
            self.model = YOLO(model_path, task='detect')
            self.model.to(device)

        self.imgsz = imgsz
        self.timing = 0

    def detect(self, img):
        start = perf_counter_ns()
        detection = self.model(img.astype('uint8'), verbose=False, imgsz=self.imgsz)
        self.timing = (perf_counter_ns()-start)/1e6
        names = detection[0].names
        result = []
        predictions = []
        confidences = []
        boxes = detection[0].boxes.data.cpu().numpy()
        for box in (boxes):
            x0, y0, x1, y1, conf, pred = box
            result.append([x0,y0, x1-x0, y1-y0])
            predictions.append(names[int(pred)])
            confidences.append(conf)

        return result, predictions, confidences #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing
    
class RTDETR_detector(BaseDetector):
    def __init__(self, model_path, imgsz, device = None):
        self.model = RTDETR(model_path)
        if device in ['gpu', 'cuda']:
            self.model.to('cuda')
        if device == 'cpu':
            self.model.to('cpu')
        self.imgsz = imgsz
        self.timing = 0

    def detect(self, img):
        start = perf_counter_ns()
        detection = self.model(img.astype('uint8'), verbose=False, imgsz=self.imgsz)
        self.timing = (perf_counter_ns()-start)/1e6
        names = detection[0].names
        result = []
        predictions = []
        confidences = []
        boxes = detection[0].boxes.data.cpu().numpy()
        for box in (boxes):
            x0, y0, x1, y1, conf, pred = box
            result.append([x0,y0, x1-x0, y1-y0])
            predictions.append(names[int(pred)])
            confidences.append(conf)

        return result, predictions, confidences #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing