from ctypes import *
import numpy as np
from time import perf_counter_ns

class Soros_detector:
    def __init__(self, lib_path, winsize=20, max_rois=50, detectionType='1D'):
        self.lib_path = lib_path
        self.winsize = winsize
        self.max_rois = max_rois
        self.detectionType = detectionType
        self.timing = 0
        self.cdll =  cdll.LoadLibrary(self.lib_path)

    def detect(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*4)()

        if self.detectionType == '1D':
            start = perf_counter_ns()
            self.cdll.sorosProcess(result, input_img, h,w, c_bool(True), c_int(self.winsize))
            self.timing = (perf_counter_ns()-start)/10e6
            results = [np.array(result)]
            return results, ['1D'], [None] #Boxes, classes, confidence scores
        elif self.detectionType == '2D':
            start = perf_counter_ns()
            self.cdll.sorosProcess(result, input_img, h,w, c_bool(False), c_int(self.winsize))
            self.timing = (perf_counter_ns()-start)/10e6
            results = [np.array(result)]
            return results, ['2D'], [None] #Boxes, classes, confidence scores
        else:
            results = []
            start = perf_counter_ns()
            self.cdll.sorosProcess(result, input_img, h,w, c_bool(True), c_int(self.winsize))
            self.timing = (perf_counter_ns()-start)/10e6
            results.append(np.array(result))
            start = perf_counter_ns()
            self.cdll.sorosProcess(result, input_img, h,w, c_bool(False), c_int(self.winsize))
            self.timing += (perf_counter_ns()-start)/10e6
            results.append(np.array(result))
            return results, ['1D','2D'], [None, None] #Boxes, classes, confidence scores
        
    def get_timing(self):
        return self.timing



