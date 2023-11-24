from ctypes import *
import numpy as np

class Soros_detector:
    def __init__(self, lib_path, winsize=20, max_rois=50, detectionType='1D'):
        self.lib_path = lib_path
        self.winsize = winsize
        self.max_rois = max_rois
        self.detectionType = detectionType
        self.cdll =  cdll.LoadLibrary(self.lib_path)

    def detect(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*4)()

        if self.detectionType == '1D':
            self.cdll.galloProcess(result, input_img, h,w, c_bool(True), c_int(self.winsize));
            results = [np.array(result)]
            return results, ['1D'], None #Boxes, classes, confidence scores
        elif self.detectionType == '2D':
            self.cdll.galloProcess(result, input_img, h,w, c_bool(False), c_int(self.winsize));
            results = [np.array(result)]
            return results, ['2D'], None #Boxes, classes, confidence scores
        else:
            results = []
            self.cdll.galloProcess(result, input_img, h,w, c_bool(True), c_int(self.winsize));
            results.append(np.array(result))
            self.cdll.galloProcess(result, input_img, h,w, c_bool(False), c_int(self.winsize));
            results.append(np.array(result))
            return results, ['1D','2D'], None #Boxes, classes, confidence scores


