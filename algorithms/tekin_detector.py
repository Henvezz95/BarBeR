from ctypes import *
import numpy as np

class Tekin_detector:
    def __init__(self, lib_path, max_rois=500):
        self.lib_path = lib_path
        self.max_rois = max_rois
        self.cdll =  cdll.LoadLibrary(self.lib_path)


    def detect_lines(self, img):
        h, w, _ = img.shape
        h,w = c_short(h), c_short(w)
        input_img = img.ctypes.data_as(POINTER(c_uint8))
        result = (c_int*(self.max_rois*4))()
        num_results = (c_int*1)()

        self.cdll.locateBarcodes(result, num_results, input_img, h,w)
        result = np.array(result).ravel()
        result = result[:int(num_results[0])*4].reshape((-1,2,2))
        return result, ['1D']*len(result), None #Boxes, classes, confidence scores