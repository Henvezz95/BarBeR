import torch
import numpy as np
from time import perf_counter_ns
from detectors_abs import BaseDetector
    

class Pytorch_detector(BaseDetector):
    def __init__(self, model_path, th=0.5, device = 'cpu'):
        if device in ['gpu', 'cuda']:
            self.model = torch.load(model_path, map_location='cuda')
        if device == 'cpu':
            self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.th = th
        self.timing = 0

    def detect(self, img):  # sourcery skip: avoid-builtin-shadow
        img_float = np.float32(img[np.newaxis,:,:,:]/255.0)
        input = torch.from_numpy(np.transpose(img_float, (0, 3, 1, 2)))

        start = perf_counter_ns()
        results = self.model(input)[0]
        self.timing = (perf_counter_ns()-start)/1e6
        boxes = results['boxes'].tensor.cpu().detach().numpy()
        predictions = results['labels'].cpu().detach().numpy()
        confidences = results['scores'].cpu().detach().numpy()
        filtered_result = list(zip(boxes,predictions, confidences))
        if filtered_result := list(
            filter(lambda x: x[-1] > self.th, filtered_result)
        ):
            boxes, classes, confidences = [x[0] for x in filtered_result], [x[1] for x in filtered_result], [x[2] for x in filtered_result]
        else:
            return [],[],[]
        boxes = [[box[0],box[1], box[2]-box[0], box[3]-box[1]] for box in boxes]
        classes = ['1D' if pred==0 else '2D' for pred in predictions]


        return boxes, classes, confidences #Boxes, classes, confidence scores
    
    def get_timing(self):
        return self.timing