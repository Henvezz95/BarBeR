import torch
import numpy as np
from time import perf_counter_ns
from detectors_abs import BaseDetector

class Detectron2_detector(BaseDetector):
    def __init__(self, model_path, th=0.5, device = 'cpu'):
        if device in ['gpu', 'cuda']:
            self.model = torch.load(model_path, map_location='cuda')
        if device == 'cpu':
            self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.th = th
        self.timing = 0

    def detect(self, img):  
        H,W,_ = img.shape
        input = [{'image':torch.from_numpy(np.transpose(img, (2, 0, 1))), 
         'height':H, 
         'width':W}]
        
        start = perf_counter_ns()
        results = self.model(input)[0]['instances'].get_fields()
        self.timing = (perf_counter_ns()-start)/1e6
        boxes = results['pred_boxes'].tensor.cpu().detach().numpy()
        predictions = results['pred_classes'].cpu().detach().numpy()
        confidences = results['scores'].cpu().detach().numpy()
        filtered_result = list(zip(boxes,predictions, confidences))
        filtered_result = list(filter(lambda x: x[-1] > self.th, filtered_result))
        if len(filtered_result) > 0:
            boxes = [x[0] for x in filtered_result]
            classes = [x[1] for x in filtered_result] 
            confidences = [x[2] for x in filtered_result]
        else:
            return [],[],[]
        boxes = [[box[0],box[1], box[2]-box[0], box[3]-box[1]] for box in boxes]
        classes = ['1D' if pred==0 else '2D' for pred in predictions]
        
        
        return boxes, classes, confidences #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing
    