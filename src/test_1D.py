import numpy as np
import cv2
import json
import sys
from tqdm import tqdm

from utils import box_to_poly, compute_1D_coverage

sys.path.append('./algorithms/') 
from gallo_detector import Gallo_detector
from zamberletti_detector import Zamberletti_detector
from yun_detector import Yun_detector
from soros_detector import Soros_detector

longest_edge_resize = 640
single_code = True
gallo_detector = Gallo_detector('./build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so')
zamberletti_detector = Zamberletti_detector('./build/Zamberletti2013/libBarcodeLibrary.so')
yun_detector = Yun_detector('./build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so')
soros_detector = Soros_detector('./build/Gallo2011-Soros2013-Yun2017/libBarcodeLocalization.so')

with open('./annotations/COCO/test.json') as json_file:
    coco_annotations = json.load(json_file)

with open('./annotations/COCO/datasets_info.json') as json_file:
    datasets_info = json.load(json_file)

ann_index = 0
image_counter = 0
minimum_area = 0
total_area = 0
Gallo_results = [0,0,0]
Zamberletti_results = [0,0,0]
Yun_results = [0,0,0]
Soros_results = [0,0,0]
for image_annotation in tqdm(coco_annotations['images']):
    id = image_annotation['id']
    file_name = image_annotation['file_name']
    true_boxes = []
    true_polygons = []
    remove_this_image = False
    while coco_annotations['annotations'][ann_index]['image_id'] == id:
        true_boxes.append(np.array(coco_annotations['annotations'][ann_index]['bbox']))
        true_polygons.append(np.array(coco_annotations['annotations'][ann_index]['segmentation']).reshape(4,2))
        if coco_annotations['annotations'][ann_index]['category_id'] != 1:
            remove_this_image = True
        
        ann_index+=1
        if ann_index >= len(coco_annotations['annotations']):
            break

    if single_code and (len(true_boxes) != 1):
        remove_this_image = True
    if remove_this_image:
        continue

    
    img_path = datasets_info['images'][file_name]['path']
    img = cv2.imread(img_path)
    image_counter+=1
    H,W,_ = img.shape
    if W > H:
        W_new = longest_edge_resize
        H_new = int(np.round((H*W_new)/W))
    else:
        H_new = longest_edge_resize
        W_new = int(np.round((W*H_new)/H))

    total_area += (W_new*H_new)/1000000

    for i in range(len(true_boxes)):
        true_boxes[i][0::2] = np.int32(np.round(W_new*true_boxes[i][0::2]/W))
        true_boxes[i][1::2] = np.int32(np.round(H_new*true_boxes[i][1::2]/H))
        true_polygons[i][:,0] = np.int32(np.round(W_new*true_polygons[i][:,0]/W))
        true_polygons[i][:,1] = np.int32(np.round(H_new*true_polygons[i][:,1]/H))
        minimum_area += (true_boxes[i][-2]* true_boxes[i][-1])/1000000 
    
    img = cv2.resize(img, (W_new, H_new), cv2.INTER_CUBIC)
    boxes, classes, confidences = gallo_detector.detect(img)
    if len(boxes) > 0:
        poly_box = box_to_poly(boxes[0])
        Gallo_results[2] += (boxes[0][-2]*boxes[0][-1])/1000000
    if compute_1D_coverage(poly_box, true_polygons[0]) > 0.9:
        Gallo_results[0]+=1
    else:
        Gallo_results[1]+=1
    
    boxes, classes, confidences = soros_detector.detect(img)
    if len(boxes) > 0:
        poly_box = box_to_poly(boxes[0])
        Soros_results[2] += (boxes[0][-2]*boxes[0][-1])/1000000
    if compute_1D_coverage(poly_box, true_polygons[0]) > 0.9:
        Soros_results[0]+=1
    else:
        Soros_results[1]+=1
    
    boxes, classes, confidences = zamberletti_detector.detect(img)
    if len(boxes) > 0:
        poly_box = box_to_poly(boxes[0])
        Zamberletti_results[2] += (boxes[0][-2]*boxes[0][-1])/1000000
    if compute_1D_coverage(poly_box, true_polygons[0]) > 0.9:
        Zamberletti_results[0]+=1
    else:
        Zamberletti_results[1]+=1

    boxes, classes, confidences = yun_detector.detect(img)

    max_score = 0
    for box in boxes:
        Yun_results[2]+=(box[-2]*box[-1])/1000000
        poly_box = box_to_poly(box)
        new_score = compute_1D_coverage(poly_box, true_polygons[0])
        if  new_score > max_score:
            max_score = new_score

    if max_score > 0.9:
        Yun_results[0]+=1
    else:
        Yun_results[1]+=1

print(image_counter, total_area, minimum_area)
print('Gallo: ', Gallo_results)
print('Zamberletti: ', Zamberletti_results)
print('Yun: ', Yun_results)
print('Soros: ', Soros_results)
