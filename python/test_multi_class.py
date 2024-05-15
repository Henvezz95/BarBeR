import numpy as np
import cv2
import json
import sys
import getopt
from tqdm import tqdm

from bounding_box import BoundingBox
from evaluators import coco_evaluator
from utils.enumerators import BBType
from utility import join, from_np
import yaml

sys.path.append('./algorithms/') 

def parse_inputs(file_path, argv):
    config_path = None
    output_path = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["cfile=", "ofolder="])
    except getopt.GetoptError:
        print(file_name, '-c <configfile> -o <outputfolder>')
        print('The configuration file must be in yaml format')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name, '-c <configfile> -o <outputfolder>')
            print('The configuration file must be in yaml format')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            config_path = arg
        elif opt in ("-o", "--ofolder"):
            output_path = arg

    if config_path == None:
        print('A configuration file in yaml format is needed to run the program')
        print(file_name, '-c <configfile> -o <outputfolder>')
        sys.exit(2)
    if output_path == None:
        print('Provide a path to save the generated Results')
        print(file_name, '-c <configfile> -o <outputfolder>')
        sys.exit(2)

    if output_path[-5:] != '.yaml':
        output_path+='.yaml'
    return config_path, output_path

def import_module(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    config_path, output_path = parse_inputs(sys.argv[0], sys.argv[1:])

    with open(config_path, "r") as stream:
       test_config = yaml.safe_load(stream)

    if "longest_edge_resize" in test_config:
        longest_edge_resize = test_config["longest_edge_resize"]
    else:
        longest_edge_resize = -1

    coco_annotation_path = test_config["coco_annotations_path"]
    if 'bins' in test_config:
        bins = test_config['bins']
    else:
        bins = []

    with open(f'{coco_annotation_path}test.json') as json_file:
        coco_annotations = json.load(json_file)

    with open(f'{coco_annotation_path}datasets_info.json') as json_file:
        datasets_info = json.load(json_file)

    detectors = {}

    for algorithm in test_config['algorithms']:
        mod = import_module(algorithm['library'])
        mod = getattr(mod, algorithm['class'])
        detectors[algorithm['name']] = mod(**algorithm['args'])

    ann_index = 0
    image_counter = 0
    GT_area = 0
    total_area = 0
    ppe_values = []
    detected_bbs = {detector_name:[] for detector_name in detectors}
    groundtruth_bbs = {detector_name:[] for detector_name in detectors}
    num_labels = {'small':0, 'medium':0, 'large':0}
    all_ppe_values = {'1D':[], '2D':[]}
    all_areas = {'1D':[], '2D':[]}
    datasets_map = {
        dataset: {
            k
            for k in datasets_info['images']
            if datasets_info['images'][k]['dataset'] == dataset
        }
        for dataset in datasets_info['datasets']
    }

    for image_annotation in tqdm(coco_annotations['images']):
        id = image_annotation['id']
        file_name = image_annotation['file_name']
        true_boxes = []
        true_polygons = []
        true_classes = []
        true_ppe = []

        img_path = datasets_info['images'][file_name]['path']
        img = cv2.imread(img_path)
        image_counter+=1
        H,W,_ = img.shape
        if longest_edge_resize > 0:
            if W > H:
                W_new = longest_edge_resize
                H_new = int(np.round((H*W_new)/W))
            else:
                H_new = longest_edge_resize
                W_new = int(np.round((W*H_new)/H))

            img = cv2.resize(img, (W_new, H_new), cv2.INTER_CUBIC)
        else:
            H_new, W_new = H, W

        total_area += (W_new*H_new)/1000000

        while coco_annotations['annotations'][ann_index]['image_id'] == id:
            true_boxes.append(np.array(coco_annotations['annotations'][ann_index]['bbox']))
            true_boxes[-1][0::2] = np.int32(np.round(W_new*true_boxes[-1][0::2]/W))
            true_boxes[-1][1::2] = np.int32(np.round(H_new*true_boxes[-1][1::2]/H))
            area = (true_boxes[-1][-2]* true_boxes[-1][-1])
            true_classes.append(coco_annotations['annotations'][ann_index]['category_id']-1)
            true_polygons.append(np.array(coco_annotations['annotations'][ann_index]['segmentation']).reshape(-1,2))
            true_polygons[-1][:,0] = np.int32(np.round(W_new*true_polygons[-1][:,0]/W))
            true_polygons[-1][:,1] = np.int32(np.round(H_new*true_polygons[-1][:,1]/H))
            ppe = float(datasets_info['images'][file_name]['ppes'][str(ann_index+1)]/W*W_new)
            true_ppe.append(ppe)
            if true_classes[-1] == 0:
                all_areas['1D'].append(area)
                all_ppe_values['1D'].append(ppe)
            else:
                all_areas['2D'].append(area)
                all_ppe_values['2D'].append(ppe)
            if area < 32**2:
                num_labels['small'] += 1
            elif area < 96**2:
                num_labels['medium'] += 1
            else:
                num_labels['large'] += 1

            GT_area += area/1000000 
            ann_index+=1
            if ann_index >= len(coco_annotations['annotations']):
                break

        dataset_name = datasets_info['images'][file_name]['dataset']

        for detector_name, detector in detectors.items():
            boxes, classes, confidences = detector.detect(img)
            detected_bbs[detector_name].extend([
                BoundingBox(file_name, 
                            0 if classes[i] == '1D' else 1, 
                            boxes[i], 
                            img_size=(W_new, H_new), 
                            confidence=confidences[i] if confidences[i] is not None else 1, 
                            bb_type=BBType.DETECTED) for i in range(len(boxes))])
            groundtruth_bbs[detector_name].extend([
                BoundingBox(file_name, 
                            true_classes[i],
                            true_boxes[i], 
                            img_size=(W_new, H_new), 
                            confidence=1, 
                            bb_type=BBType.GROUND_TRUTH) for i in range(len(true_boxes))])

    num_labels['all'] = num_labels['small']+num_labels['medium']+num_labels['large']

    results = {"image_count": int(image_counter), 
               "num_labels":num_labels, 
               "total_area": float(total_area), 
               "GT_area": float(GT_area), 
               "evaluation":{},
               "ppe_evaluation":{}}

    for detector_name in detectors:
        COCOevaluation = coco_evaluator.get_coco_summary2(groundtruth_bbs[detector_name], detected_bbs[detector_name])
        results['evaluation'][detector_name] = {key:{k:from_np(v) for k,v in COCOevaluation[key].items()} for key in COCOevaluation}
        ppe_evaluation = coco_evaluator.get_pixel_density_summary(groundtruth_bbs[detector_name], detected_bbs[detector_name], bins)
        results["ppe_evaluation"][detector_name] = ppe_evaluation

    with open(output_path, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, sort_keys=False)
        


    with open(output_path, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, sort_keys=False)