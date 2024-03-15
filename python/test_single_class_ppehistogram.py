import numpy as np
import cv2
import json
import sys
import getopt
from tqdm import tqdm

from utility import from_np
from bounding_box import BoundingBox
from evaluators import coco_evaluator
from utils.enumerators import BBType
import yaml

sys.path.append('./algorithms/') 
bins = [-100, 0, 1,2,3,4,5,6,7,100]

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
    single_code = test_config["single_ROI"]
    class_id = 1 if test_config["class"] == '1D' else 2

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
    detected_bbs = {detector_name:[] for detector_name in detectors}
    groundtruth_bbs = {detector_name:[] for detector_name in detectors}
    num_labels = {'small':0, 'medium':0, 'large':0}
    all_ppe_values = []
    all_areas = []
    datasets_map = {dataset:set(k for k in datasets_info['images'] if datasets_info['images'][k]['dataset']==dataset)
                           for dataset in datasets_info['datasets']}

    
    for image_annotation in tqdm(coco_annotations['images']):
        id = image_annotation['id']
        file_name = image_annotation['file_name']
        true_boxes = []
        true_polygons = []
        true_classes = []
        true_ppe = []
        remove_this_image = False
        while coco_annotations['annotations'][ann_index]['image_id'] == id:
            if coco_annotations['annotations'][ann_index]['category_id'] != class_id:
                remove_this_image = True

            true_boxes.append(np.array(coco_annotations['annotations'][ann_index]['bbox']))
            true_classes.append(coco_annotations['annotations'][ann_index]['category_id']-1)
            true_polygons.append(np.array(coco_annotations['annotations'][ann_index]['segmentation']).reshape(-1,2))
            ppe = float(datasets_info['images'][file_name]['ppes'][str(ann_index+1)])
            true_ppe.append(ppe)
            ann_index+=1
            if ann_index >= len(coco_annotations['annotations']):
                break
            
        if remove_this_image:
            continue  
        if single_code and len(true_boxes) != 1:
            continue    

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
         
        total_area += (W_new*H_new)/10e6        
           
        for i in range(len(true_boxes)):
            true_boxes[i][0::2] = np.int32(np.round(W_new*true_boxes[-1][0::2]/W))
            true_boxes[i][1::2] = np.int32(np.round(H_new*true_boxes[-1][1::2]/H))
            true_polygons[-1][:,0] = np.int32(np.round(W_new*true_polygons[-1][:,0]/W))
            true_polygons[-1][:,1] = np.int32(np.round(H_new*true_polygons[-1][:,1]/H))
            area = (true_boxes[i][-2]* true_boxes[i][-1])
            true_ppe[i] = true_ppe[i]/W*W_new
            all_areas.append(area)
            all_ppe_values.append(true_ppe[i])
            GT_area += area/10e6

            if area < 32**2:
                num_labels['small'] += 1
            elif area < 96**2:
                num_labels['medium'] += 1
            else:
                num_labels['large'] += 1
        
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
                            ppe = true_ppe[i],
                            bb_type=BBType.GROUND_TRUTH) for i in range(len(true_boxes))])


    print(image_counter, num_labels, total_area, GT_area)
    results = {"image_count": int(image_counter), 
               "num_labels":num_labels, 
               "total_area": float(total_area), 
               "GT_area": float(GT_area), 
               "bins": [],
               "evaluation":{},
               "ppe_evaluation": {},
               "single dataset_evaluations":{}}
    
    for detector_name in detectors:
        COCOevaluation = coco_evaluator.get_coco_summary2(groundtruth_bbs[detector_name], detected_bbs[detector_name])
        results['evaluation'][detector_name] = {key:{k:from_np(v) for k,v in COCOevaluation[key].items()} for key in COCOevaluation}
        ppe_evaluation = {}
        for k in range(len(bins)-1):
            file_names = set(bbs._image_name for bbs in groundtruth_bbs[detector_name] if bins[k] <= bbs._ppe < bins[k+1])
            filtered_gt = [bbs for bbs in groundtruth_bbs[detector_name] if bbs._image_name in file_names]
            filtered_dt = [bbs for bbs in detected_bbs[detector_name] if bbs._image_name in file_names]
            if len(filtered_gt) > 0:
                COCOevaluation = coco_evaluator.get_coco_summary2(filtered_gt, filtered_dt)
                ppe_evaluation[str(bins[k])+'_'+str(bins[k+1])] = {key:{k:from_np(v) for k,v in COCOevaluation[key].items()} for key in COCOevaluation}
        
        results["ppe_evaluation"][detector_name] = ppe_evaluation
        results["single dataset_evaluations"][detector_name] = {}
        for dataset_name, dataset_images in datasets_map.items():
            filtered_gt = [bbs for bbs in groundtruth_bbs[detector_name] if bbs._image_name in dataset_images]
            filtered_dt = [bbs for bbs in detected_bbs[detector_name] if bbs._image_name in dataset_images]
            if len(filtered_gt) > 0:
                COCOevaluation = coco_evaluator.get_coco_summary2(filtered_gt, filtered_dt)
                results['single dataset_evaluations'][detector_name][dataset_name] = {key:{k:from_np(v) for k,v in COCOevaluation[key].items()} for key in COCOevaluation}
    
    
    with open(output_path, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, sort_keys=False)