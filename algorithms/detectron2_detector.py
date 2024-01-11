import torch
import numpy as np
from detectron2.config import get_cfg

class Detectron2_detector:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def detect(self, img):
        H,W,_ = img.shape
        input = [{'image':torch.from_numpy(np.transpose(img, (2, 0, 1))), 
         'height':H, 
         'width':W}]
        
        results = self.model(input)[0]['instances'].get_fields()
        boxes = results['pred_boxes'].tensor.cpu().detach().numpy()
        boxes = [[box[0],box[1], box[2]-box[0], box[3]-box[1]] for box in boxes]
        predictions = results['pred_classes'].cpu().detach().numpy()
        classes = ['1D' if pred==0 else '2D' for pred in predictions]
        confidences = results['scores'].cpu().detach().numpy()
        
        return boxes, classes, confidences #Boxes, classes, confidence scores