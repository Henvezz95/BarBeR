import math


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
