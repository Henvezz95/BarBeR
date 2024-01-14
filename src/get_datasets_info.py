import json
import cv2
import hashlib
import os
import datetime
from tqdm import tqdm
from glob import glob
import numpy as np

config_file_path = './config/generate_coco_annotations_config.json'

with open(config_file_path) as json_file:
    annotations_config = json.load(json_file)

vgg_annotations_basepath = annotations_config['vgg_annotations_path']
img_paths = glob(annotations_config['images_path']+'*.[jJ][pP][gGeE]*')

if len(annotations_config['annotation_files']) == 0:
    annotations = glob(f'{vgg_annotations_basepath}*.json', recursive=True)
else:
    annotations = [vgg_annotations_basepath + file_path for file_path in annotations_config['annotation_files']]

annotations = sorted(annotations)
total_images = 0
total_1D = 0
total_2D = 0
all_sizes = []
for annotation_path in annotations:
    num_1D = 0
    num_2D = 0
    sizes = []

    with open(annotation_path) as json_file:
        metadata = json.load(json_file)
    for key in metadata['_via_img_metadata']:
        sizes.append([metadata['_via_img_metadata'][key]['size'], metadata['_via_img_metadata'][key]['filename']])
        for region in metadata['_via_img_metadata'][key]['regions']:
            if region['region_attributes']['Type'] in ['QR', 'DATAMATRIX', 'AZTEC', 'PDF417']:
                if os.path.basename(annotation_path) == 'Muenster.json':
                    print(metadata['_via_img_metadata'][key]['filename'])
                num_2D+= 1
            else:
                num_1D += 1
    total_1D += num_1D
    total_2D += num_2D
    total_images += len(metadata['_via_img_metadata'])
    all_sizes.extend(sizes)

    min_size = sorted(sizes, key=lambda x: x[0])[0]
    max_size = sorted(sizes, key=lambda x: x[0])[-1]
    img_min = cv2.imread("./dataset/images/"+min_size[1])
    H_min, W_min, _ = img_min.shape
    img_max = cv2.imread("./dataset/images/"+max_size[1])
    H_max, W_max, _ = img_max.shape
    print(os.path.basename(annotation_path), '&', len(metadata['_via_img_metadata']), '&', str(W_min)+'\\times'+str(H_min), '&',  str(W_max)+'\\times'+str(H_max), '&', num_1D, '&', num_2D)

print('\n')
min_size = sorted(all_sizes, key=lambda x: x[0])[0]
max_size = sorted(all_sizes, key=lambda x: x[0])[-1]
img_min = cv2.imread("./dataset/images/"+min_size[1])
H_min, W_min, _ = img_min.shape
img_max = cv2.imread("./dataset/images/"+max_size[1])
H_max, W_max, _ = img_max.shape
print('Total', '&', total_images, '&', str(W_min)+'\\times'+str(H_min), '&',  str(W_max)+'\\times'+str(H_max), '&', total_1D, '&', total_2D)