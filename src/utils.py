import math
import numpy as np
import cv2


class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y



def GetAreaOfPolyGon(points_x, points_y):
    points = []
    for index in range(len(points_x)):
        points.append(Point(points_x[index], points_y[index]))
    area = 0
    if(len(points)<3):
        
         raise Exception("error")

    p1 = points[0]
    for i in range(1,len(points)-1):
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
    area = math.sqrt(area)
    return area

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
