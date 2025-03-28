import math
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F

def tensor_resize(images: torch.Tensor, new_h: int, new_w: int, mode: str = 'bicubic') -> torch.Tensor:
    """
    Resizes a batch of images from (N, C, H, W) to (N, C, new_h, new_w) using bicubic interpolation.
    """
    # If your tensor is not float, convert it:
    if not images.is_floating_point():
        images = images.float()

    return F.interpolate(
        images,
        size=(new_h, new_w),
        mode=mode,
        align_corners=False
    )


class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

def from_np(value):
    if value is None:
        return 'nan'
    if isinstance(value, list):
        if not value:
            return 'nan'
        return 'nan' if value[0] is None else value
    if np.isnan(value):
        return 'nan'
    return value.item() if hasattr(value, 'item') else value

def GetAreaOfPolyGon(points_x, points_y):
    points = []
    points.extend(
        Point(points_x[index], points_y[index])
        for index in range(len(points_x))
    )
    area = 0
    if(len(points)<3):

         raise Exception("error")

    p1 = points[0]
    for _ in range(1,len(points)-1):
        p2 = points[1]
        p3 = points[2]


        vecp1p2 = Point(p2.x-p1.x,p2.y-p1.y)
        vecp2p3 = Point(p3.x-p2.x,p3.y-p2.y)



        vecMult = vecp1p2.x*vecp2p3.y - vecp1p2.y*vecp2p3.x
        sign = 0
        if(vecMult>0):
            sign = 1
        elif(vecMult<0):
            sign = -1

        triArea = GetAreaOfTriangle(p1,p2,p3)*sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1,p2,p3):
    area = 0
    p1p2 = GetLineLength(p1,p2)
    p2p3 = GetLineLength(p2,p3)
    p3p1 = GetLineLength(p3,p1)
    s = (p1p2 + p2p3 + p3p1)/2
    area = s*(s-p1p2)*(s-p2p3)*(s-p3p1)
    return math.sqrt(area)

def GetLineLength(p1,p2):

    length = math.pow((p1.x-p2.x),2) + math.pow((p1.y-p2.y),2) 
    length = math.sqrt(length)   
    return length 


def get_segmenation(coord_x, coord_y):
    seg = []
    for x, y in zip(coord_x, coord_y):
        seg.extend((x, y))
    return [seg]   

def box_to_poly(box):
    if len(box) == 0:
        return []
    x,y,w,h = box
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])

def intersect_area(box1, box2, format='xywh'):
    if format=='xywh':
        box1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
        box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    return (xB - xA) * (yB - yA)

def compute_1D_coverage(prediction, true_polygon):
    if len(prediction) == 0:
        return 0
    contours = np.float32(np.array([true_polygon[0], true_polygon[1], true_polygon[2], true_polygon[3]]))
    pts = np.float32([[0, 1], [0, 0], [1, 0], [1, 1]])
    M = cv2.getPerspectiveTransform(contours, pts)

    transformed_prediction = cv2.perspectiveTransform(np.float32(prediction).reshape(-1,1,2),M)
    minX = np.min(transformed_prediction[:,0])
    maxX = np.max(transformed_prediction[:,0])
    return np.clip(maxX,0,1)-np.clip(minX,0,1)

def get_contours_and_boxes(binarized_map, min_area=0):
    assert binarized_map.dtype == np.uint8
    contours, _ = cv2.findContours(
        binarized_map,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours = list(filter(lambda cnt: cv2.contourArea(cnt) > min_area, contours))
    boxes = np.array([cv2.boundingRect(cnt) for cnt in contours])

    return contours, boxes

def torch_load(model_path, map_location="cpu"):
    """
    Automatically detects if the model is a standard PyTorch model or a TorchScript model 
    and loads it accordingly.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    try:
        # Attempt to load as a TorchScript model
        model = torch.jit.load(model_path, map_location=map_location)
        print(f"Loaded TorchScript model from {model_path}")
    except RuntimeError:
        # If it fails, try loading as a standard PyTorch model
        model = torch.load(model_path, map_location=map_location)
        print(f"Loaded standard PyTorch model from {model_path}")

    return model
