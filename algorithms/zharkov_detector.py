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

class Zharkov_detector(BaseDetector):
    def __init__(self, model_path, th=0.5, minArea=10, device = 'gpu'):
        self.model = torch.load(model_path)
        # First layer weights are scaled to operate in the range 0..255 instead of 0..1
        # This is done to increase the pre-processing speed
        if device in ['gpu', 'cuda']:
            self.model.cuda()
        if device == 'cpu':
            self.model.cpu()
        
        self.model.net[0][0].weight.data = self.model.net[0][0].weight.data/255.0 
        self.th = th
        self.minArea = minArea
        self.device = device
        self.timing = 0

    def detect(self, img):
        start = perf_counter_ns()
        # Pre-processing
        img = np.transpose(img, (2,0,1)).astype('float32')
        img = torch.from_numpy(img)
        
        # Model Prediction
        if self.device in ['gpu', 'cuda']:
            heatmap = self.model(img.cuda())
        else:
            heatmap = self.model(img)
        
        # Heatmap Processing
        heatmap[0,:,:] = torch.sigmoid(heatmap[0,:,:])
        heatmap[1:,:,:] = torch.nn.Softmax(dim=0)(heatmap[1:,:,:])
        detection_heatmap = heatmap.detach().cpu().numpy()[0,:,:]
        classification_heatmap = np.transpose(heatmap.detach().cpu().numpy(), (1,2,0))[:,:,1:]
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
