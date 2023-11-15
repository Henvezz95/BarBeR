from ctypes import *
import numpy as np
import cv2

class zamberletti_detector:
    def __init__(self, lib_path, winsize=20, max_rois=50):
        self.lib_path = lib_path
        self.winsize = winsize
        self.max_rois = max_rois
        self.cdll =  cdll.LoadLibrary(self.lib_path)

    def detect(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*self.max_rois*8)()
        num_results = (c_int*1)()
        angle = (c_double*1)()

        self.cdll.locateBarcode(result, angle, num_results, input_img, h,w);
        result = np.array(result)[:num_results[0]*8].reshape((-1,4, 2))
        
        # Rotate by the computed angle
        for i in range(len(result)):
            center = np.mean(result[i], axis=0)
            M = cv2.getRotationMatrix2D(center, -angle[0], 1.0)
            result[i] = np.matmul(result[i], M[:,:2].T)+M[:,2]

        return result, ['1D']*len(result), None #Boxes, classes, confidence scores

