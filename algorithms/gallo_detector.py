from ctypes import *
import numpy as np
from time import perf_counter_ns

class Gallo_detector:
    def __init__(self, lib_path, winsize=20, max_rois=50):
        self.lib_path = lib_path
        self.winsize = winsize
        self.max_rois = max_rois
        self.timing = 0
        self.cdll =  cdll.LoadLibrary(self.lib_path)

    def detect(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*4)()

        start = perf_counter_ns()
        self.cdll.galloProcess(result, input_img, h,w, c_int(self.winsize))
        self.timing = (perf_counter_ns()-start)/1e6
        result = np.array(result)
        return [result], ['1D'], [None] #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing

