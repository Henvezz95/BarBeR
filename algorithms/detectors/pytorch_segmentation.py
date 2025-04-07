import torch
import numpy as np
import cv2
from time import perf_counter_ns
import sys
from detectors_abs import BaseDetector


sys.path.append('./python/') 
sys.path.append('./BaFaLo/') 

from utils.utility import get_contours_and_boxes
from utils.activations import act_dict
from utils.model_wrapper import ModelWrapper

def smooth_binary(img_binary, kernel_size, threshold):
    result = cv2.boxFilter(img_binary, -1, (kernel_size, kernel_size), normalize=False)
    result = cv2.threshold(result, threshold, 1, cv2.THRESH_BINARY)[1]
    return result

class Pytorch_segmenter(BaseDetector):
    def __init__(self, 
                 model_path, 
                 th=0.5, 
                 minArea=350, 
                 gray_scale = True,
                 clahe=False, 
                 device = 'cuda',
                 num_threads = 1,
                 single_class = False,
                 activation = 'sigmoid',
                 remove_first_channel = False,
                 onnx = False,
                 class_type = '1D',
                 min_input_size = 0):
        
        # First layer weights are scaled to operate in the range 0..255 instead of 0..1
        # This is done to increase the pre-processing speed
        #self.model.net[0][0].weight.data = self.model.net[0][0].weight.data/255.0 
        self.th = th
        self.gray_scale = gray_scale
        self.clahe = clahe
        self.minArea = minArea
        self.device = device
        self.timing = 0
        self.single_class = single_class
        self.class_type = class_type
        self.min_input_size = min_input_size
        self.remove_first_channel = remove_first_channel
        self.model = ModelWrapper(model = torch.load(model_path, weights_only=False),
                                  num_threads = num_threads,
                                  device = device,
                                  activation = activation)
        if onnx:
           self.model.convert2onnx()


    def _get_heatmap(self, img):
        H,W,_ = img.shape
        a = max(int(np.ceil(H/32)*32), self.min_input_size ) - H
        b = max(int(np.ceil(W/32)*32), self.min_input_size ) - W
        img = np.pad(img, ((0,a),(0,b),(0,0)))
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
                img = clahe.apply(np.uint8(img))
                img = np.float32(img)
            img = img[:,:,np.newaxis]

        img = img/255.0
        img = np.transpose(img, (2,0,1)).astype('float32')
        return self.model.forward(img[np.newaxis])[0]
    
    def segment(self,img):
        start = perf_counter_ns()
        heatmap = self._get_heatmap(img)
        result = np.uint8(heatmap > self.th)
        self.timing = (perf_counter_ns()-start)/1e6
        return result
    
    def detect(self, img):
        start = perf_counter_ns()
        H,W,_ = img.shape
        heatmap = self._get_heatmap(img)
        # Post-processing
        if self.single_class:
            idx = 1 if self.remove_first_channel else 0
            heatmap_binarized = np.uint8(heatmap[idx] > self.th)
            _, boxes = get_contours_and_boxes(heatmap_binarized, min_area=self.minArea)
            return boxes, [self.class_type]*len(boxes), [None]*len(boxes)

        if self.remove_first_channel:
            heatmap_1D, heatmap_2D = heatmap[1], heatmap[2]
            heatmap_1D = cv2.resize(heatmap_1D, (W,H), interpolation=cv2.INTER_CUBIC)
            heatmap_2D = cv2.resize(heatmap_2D, (W,H), interpolation=cv2.INTER_CUBIC)
        else:
            heatmap_1D, heatmap_2D = heatmap[0], heatmap[1]
        heatmap_1D_binarized = np.uint8(heatmap_1D > self.th)
        contours_1D, boxes_1D = get_contours_and_boxes(heatmap_1D_binarized, min_area=self.minArea)
        heatmap_2D_binarized = np.uint8(heatmap_2D > self.th)
        contours_2D, boxes_2D = get_contours_and_boxes(heatmap_2D_binarized, min_area=self.minArea)

        # Generate classifications and confidences 
        H,W = heatmap_1D_binarized.shape
        boxes = []
        predictions = []
        confidences = []
        for idx, contour in enumerate(contours_1D):
            mask = np.zeros((H,W), np.uint8)
            cv2.fillConvexPoly(mask, points=contour, color=[1])
            boxes.append(boxes_1D[idx])
            predictions.append('1D')
            confidences.append(cv2.mean(heatmap_1D, mask=mask)[0])
        for idx, contour in enumerate(contours_2D):
            mask = np.zeros((H,W), np.uint8)
            cv2.fillConvexPoly(mask, points=contour, color=[1])
            boxes.append(boxes_2D[idx])
            predictions.append('2D')
            confidences.append(cv2.mean(heatmap_2D, mask=mask)[0])

        self.timing = (perf_counter_ns()-start)/1e6
        return boxes, predictions, confidences  #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing

class BaFaLo_2step_detector(BaseDetector):
    def __init__(self, model_path_coarse, 
                 model_path_fine, 
                 th1=0.2, th2=0.5, 
                 minArea1=100, 
                 minArea2=400,
                 tflite=False, 
                 device = 'cpu',
                 single_class=True):
        self.bafalo_coarse = Pytorch_segmenter(model_path_coarse, 
                                             th=th1, 
                                             minArea=minArea1, 
                                             clahe=False, 
                                             device = device,
                                             single_class=single_class)
        self.minArea = minArea1*4
        self.bafalo_fine = Pytorch_segmenter(model_path_fine, 
                                             th=th2, 
                                             minArea=minArea2, 
                                             clahe=False, 
                                             device = device)
    
    def detect(self, img):
        start = perf_counter_ns()
        boxes_coarse, _, _ = self.bafalo_coarse.detect(img[::2,::2])
        boxes = []
        predictions = []
        confidences = []
        H,W,_ = img.shape
        for box in boxes_coarse:
            x,y,w,h = box*2
            x0 = max(x,0)
            x1 = min(x+w, W)
            y0 = max(y, 0)
            y1 = min(y+h, H)
            if (x1-x0)*(y1-y0) > self.minArea:
                out = self.bafalo_fine.detect(img[y0:y1, x0:x1])
                boxes_fine = [a+[x,y,0,0] for a in out[0]]
                boxes.extend(boxes_fine)
                predictions.extend(out[1])
                confidences.extend(out[2])
        self.timing = (perf_counter_ns()-start)/1e6
        return boxes, predictions, confidences  #Boxes, classes, confidence scores

    def get_timing(self):
        return self.timing
