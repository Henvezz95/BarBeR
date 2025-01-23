import torch
import numpy as np
import cv2
from time import perf_counter_ns
import sys
from detectors_abs import BaseDetector

sys.path.append('./python/') 
sys.path.append('./BaFaLo/') 

from arch import *
from utils.utility import get_contours_and_boxes
from utils.converters import pytorch2tflite

def smooth_binary(img_binary, kernel_size, threshold):
    result = cv2.boxFilter(img_binary, -1, (kernel_size, kernel_size), normalize=False)
    result = cv2.threshold(result, threshold, 1, cv2.THRESH_BINARY)[1]
    return result

class BaFaLo_detector(BaseDetector):
    def __init__(self, 
                 model_path, 
                 th=0.5, 
                 minArea=350, 
                 clahe=False, 
                 tflite=False, 
                 quantize=False,
                 device = 'cpu',
                 single_class = False,
                 class_type = '1D'):
        self.model = torch.load(model_path, map_location='cpu')
        if tflite:
            self.interpreter = pytorch2tflite(self.model, 
                                               tmp_folder='./tmp/', 
                                               input_shape=(1,1,512,640), 
                                               quantize = quantize,
                                               num_threads=1)
            self.tflite = True
        else:
            self.tflite = False
        # First layer weights are scaled to operate in the range 0..255 instead of 0..1
        # This is done to increase the pre-processing speed
        if device in ['gpu', 'cuda']:
            self.model.to('cuda')
        if device == 'cpu':
            self.model.to('cpu')
        
        #self.model.net[0][0].weight.data = self.model.net[0][0].weight.data/255.0 
        self.th = th
        self.clahe = clahe
        self.minArea = minArea
        self.device = device
        self.timing = 0
        self.single_class = single_class
        self.class_type = class_type

    def detect(self, img):
        start = perf_counter_ns()
        # Pre-processing
        H,W,c = img.shape
        a = int(np.ceil(H/32)*32)-H
        b = int(np.ceil(W/32)*32)-W
        img = np.pad(img, ((0,a),(0,b),(0,0)))
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[:,:,0]
        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            img = clahe.apply(np.uint8(img))
            img = np.float32(img)
        img = img[:,:,np.newaxis]/255.0
        img = np.transpose(img, (2,0,1)).astype('float32')
        if self.tflite:
            heatmap = self._run_tflite_interpreter(img[np.newaxis])
            heatmap = torch.from_numpy(heatmap)[0]
        else:
            img = torch.from_numpy(img[np.newaxis])
            # Model Prediction
            if self.device in ['gpu', 'cuda']:
                heatmap = self.model(img.cuda())[0]
            else:
                heatmap = self.model(img)[0]

        # Heatmap Processing
        heatmap = torch.sigmoid(heatmap).detach().cpu().numpy()
        if self.single_class:
            heatmap_binarized = np.uint8(heatmap[0] > self.th)
            _, boxes = get_contours_and_boxes(heatmap_binarized, min_area=self.minArea)
            return boxes, [self.class_type]*len(boxes), [None]*len(boxes)
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

    # TODO Rename this here and in `detect`
    def _run_tflite_interpreter(self, img):
        # Get input and output tensors
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # Allocate tensors
        self.interpreter.resize_tensor_input(input_details[0]['index'], img.shape, strict=True)
        self.interpreter.allocate_tensors()
        # Test the model on random input data
        self.interpreter.set_tensor(input_details[0]['index'], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(output_details[0]['index'])
    
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
        self.bafalo_coarse = BaFaLo_detector(model_path_coarse, 
                                             th=th1, 
                                             minArea=minArea1, 
                                             clahe=False, 
                                             tflite=tflite, 
                                             device = device,
                                             single_class=single_class)
        self.minArea = minArea1*4
        self.bafalo_fine = BaFaLo_detector(model_path_fine, 
                                             th=th2, 
                                             minArea=minArea2, 
                                             clahe=False, 
                                             tflite=tflite, 
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
