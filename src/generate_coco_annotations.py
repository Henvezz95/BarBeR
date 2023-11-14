import json
import cv2
import hashlib
import os
from utils import GetAreaOfPolyGon, get_segmenation
import datetime
from tqdm import tqdm
from glob import glob
import numpy as np

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(annotation_id, image_id, category_id, is_crowd,
                           area, bounding_box, segmentation):

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,
        "bbox": bounding_box,# [x,y,width,height]
        "segmentation": segmentation# [polygon]
    }

    return annotation_info

def convert_all(img_files, annpaths):
    paths_dictionary = {os.path.basename(path):path for path in img_files}

    coco_output = {
        'licenses': [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by/4.0/",
            }
        ],
        'categories': [
            {
                'id': 1,
                'name': '1D',
                'supercategory': 'barcode',
            },
            {
                'id': 2,
                'name': '2D',
                'supercategory': 'barcode',
            },
        ],
        'images': [],
        'annotations': [],
        'info': {
            "description": 'Barcode Dataset',
            "url": "",
            "version": "0.1.0",
            "year": datetime.date.today().strftime("%Y"),
            "contributor": "Enrico Vezzali",
            "date_created": datetime.datetime.utcnow().isoformat(' '),
        },
    }

    iscrowd = 0
    # annotations id start from zero
    ann_id = 1
    img_id = 1
    file_names = []
    for annpath in tqdm(annpaths):
        ann = json.load(open(annpath))
        ann = ann['_via_img_metadata']
        #in VIA annotations, keys are image name
        for _, key in enumerate(ann.keys()):
            filename = ann[key]['filename']
            if filename not in paths_dictionary.keys():
                continue
            if filename in file_names:
                continue
            else:
                file_names.append(filename)
            img = cv2.imread(paths_dictionary[filename])
            # make image info and storage it in coco_output['images']
            H,W = img.shape[:2]
            image_info = create_image_info(img_id, os.path.basename(filename), [W,H])
            coco_output['images'].append(image_info)
            regions = ann[key]["regions"]
            # for one image ,there are many regions,they share the same img id
            for region in regions:
                cat = region['region_attributes']['Type']
                cat_id = 2 if cat in ['QR', 'DATAMATRIX', 'AZTEC', 'PDF417'] else 1
                if region['shape_attributes']['name'] == 'polygon':
                    points_x = region['shape_attributes']['all_points_x']
                    points_y = region['shape_attributes']['all_points_y']
                elif region['shape_attributes']['name'] == 'rect':
                    x = region.get("shape_attributes").get("x")
                    y = region.get("shape_attributes").get("y")
                    w = region.get("shape_attributes").get("width")
                    h = region.get("shape_attributes").get("height")
                    points_x = [x, x + w, x + w, x]
                    points_y = [y, y, y + h, y + h]
                area = GetAreaOfPolyGon(points_x, points_y)
                min_x = min(points_x)
                max_x = max(points_x)
                min_y = min(points_y)
                max_y = max(points_y)
                box = [min_x, min_y, max_x-min_x, max_y-min_y]
                segmentation = get_segmenation(points_x, points_y)
                # make annotations info and storage it in coco_output['annotations']
                ann_info = create_annotation_info(ann_id, img_id, cat_id, iscrowd, area, box, segmentation)
                coco_output['annotations'].append(ann_info)
                ann_id = ann_id + 1
            img_id += 1

    return coco_output


config_file_path = './config/generate_coco_annotations_config.json'

with open(config_file_path) as json_file:
    annotations_config = json.load(json_file)

vgg_annotations_basepath = annotations_config['vgg_annotations_path']
img_paths = glob(annotations_config['images_path']+'*.[jJ][pP][gGeE]*')
output_path = annotations_config['output_path']
train_val_test_split = annotations_config['train_val_test_split']
salt = annotations_config['salt']

if len(annotations_config['annotation_files']) == 0:
    annotations = glob(f'{vgg_annotations_basepath}*.json', recursive=True)
else:
    annotations = [vgg_annotations_basepath + file_path for file_path in annotations_config['annotation_files']]


train_val_test_split = np.array(train_val_test_split)/np.sum(train_val_test_split)
th1 = train_val_test_split[0]
th2 = th1+train_val_test_split[1]

train_files = []
val_files = []
test_files = []

for img_path in img_paths:
    file_name = os.path.basename(img_path)
    hash_code = int(hashlib.sha256(((file_name+salt).encode('utf-8'))).hexdigest(), 16) % 10 ** 4
    if hash_code < th1 * (10 ** 4):
        train_files.append(img_path)
    elif hash_code < th2 * (10 ** 4):
        val_files.append(img_path) 
    else:
        test_files.append(img_path)
print('Train Images: '+str(len(train_files)), 
      'Validation Images: '+str(len(val_files)),
      'Test Images: '+str(len(test_files)))

coco_train = convert_all(train_files, annotations)
with open(output_path + 'train.json', "w") as outfile: 
    json.dump(coco_train, outfile, indent=2)

coco_val = convert_all(val_files, annotations)
with open(output_path + 'val.json', "w") as outfile: 
    json.dump(coco_val, outfile, indent=2)

coco_test = convert_all(test_files, annotations)
with open(output_path + 'test.json', "w") as outfile: 
    json.dump(coco_test, outfile, indent=2)









