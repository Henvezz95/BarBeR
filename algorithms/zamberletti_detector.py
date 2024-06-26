from ctypes import *
import numpy as np
import cv2
from time import perf_counter_ns
from detectors_abs import BaseDetector

class Zamberletti_detector(BaseDetector):
    def __init__(self, lib_path, net_path='./Zamberletti2013/net61x3.net', winsize=20, max_rois=50):
        self.lib_path = lib_path
        self.winsize = winsize
        self.max_rois = max_rois
        self.timing = 0
        self.cdll =  cdll.LoadLibrary(self.lib_path)
        self.cdll.initialize(net_path.encode('utf-8'))

    def detect_polygons(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*self.max_rois*8)()
        num_results = (c_int*1)()
        angle = (c_double*1)()

        start = perf_counter_ns()
        self.cdll.locateBarcode(result, angle, num_results, input_img, h,w)
        self.timing = (perf_counter_ns()-start)/1e6
        result = np.array(result).ravel()
        result = result[:int(num_results[0])*8].reshape((-1,4, 2))
        
        # Rotate by the computed angle
        for i in range(len(result)):
            center = np.mean(result[i], axis=0)
            M = cv2.getRotationMatrix2D(center, -angle[0], 1.0)
            result[i] = np.matmul(result[i], M[:,:2].T)+M[:,2]

        return result, ['1D']*len(result), [None]*len(result) #Polygons, classes, confidence scores

    def detect(self, img):
        polygons, classes, confidences = self.detect_polygons(img)
        result = []
        for poly in polygons:
            result.append(cv2.boundingRect(poly))

        return result, classes, confidences #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing

    


