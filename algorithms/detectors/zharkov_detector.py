import torch
import numpy as np
import cv2
from time import perf_counter_ns
import sys
from detectors_abs import BaseDetector

sys.path.append('./python/') 
sys.path.append('./Zharkov2019/') 
from models import ZharkovDilatedNet
from utils.utility import get_contours_and_boxes
from utils.model_wrapper import ModelWrapper

class Zharkov_detector(BaseDetector):
    def __init__(self, 
                 model_path, 
                 th=0.5, 
                 minArea=100, 
                 onnx = False, 
                 device = 'cpu', 
                 num_threads=1):
        model = torch.load(model_path)
        model.net[0][0].weight.data = model.net[0][0].weight.data/255.0 
        self.model = ModelWrapper(model = model,
                                  num_threads = num_threads,
                                  device = device,
                                  activation = 'linear')
        if onnx:
           self.model.convert2onnx()
        
        self.th = th
        self.minArea = minArea
        self.device = device
        self.timing = 0

    def _get_heatmap(self, img):
        img = np.transpose(img, (2,0,1)).astype('float32')
        heatmap = torch.from_numpy(self.model.forward(img[np.newaxis]))[0]

        # Heatmap Processing
        heatmap[0,:,:] = torch.sigmoid(heatmap[0,:,:])
        heatmap[1:,:,:] = torch.nn.Softmax(dim=0)(heatmap[1:,:,:])
        return heatmap.detach().cpu().numpy()

    def detect(self, img):
        start = perf_counter_ns()
        # Pre-processing
        heatmap = self._get_heatmap(img)
        detection_heatmap = heatmap[0,:,:]
        classification_heatmap = np.transpose(heatmap, (1,2,0))[:,:,1:]
        binarized_map = np.uint8(detection_heatmap>self.th)
        contours, boxes = get_contours_and_boxes(binarized_map, min_area=self.minArea)
        boxes = np.array(boxes)*4

        # Generate classifications and confidences 
        H,W,c = classification_heatmap.shape
        predictions = []
        confidences = []
        for contour in contours:
            mask = np.zeros((H,W), np.uint8)
            cv2.fillConvexPoly(mask, points=contour, color=[1])
            mean = cv2.mean(classification_heatmap, mask=mask)[:c]
            if mean[0] > mean[1]:
                predictions.append('1D')
            else:
                predictions.append('2D')
            confidences.append(cv2.mean(detection_heatmap, mask=mask)[0])
        
        self.timing = (perf_counter_ns()-start)/1e6
        return boxes, predictions, confidences  #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing
