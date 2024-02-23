import numpy as np
import cv2
import json
import sys
import getopt
from tqdm import tqdm
import torch

from utility import from_np
from bounding_box import BoundingBox
from evaluators import coco_evaluator
from utils.enumerators import BBType
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
    NUM_REPEATS = test_config["num_repeats"] if "num_repeats" in test_config else 3
    num_threads = test_config["num_threads"] if "num_threads" in test_config else 1
    step = test_config["step"] if "step" in test_config else 1
    
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    with open(f'{coco_annotation_path}datasets_info.json') as json_file:
        datasets_info = json.load(json_file)
    
    detectors = {}

    for algorithm in test_config['algorithms']:
        mod = import_module(algorithm['library'])
        mod = getattr(mod, algorithm['class'])
        detectors[algorithm['name']] = mod(**algorithm['args'])

    total_area = 0
    image_count = 0
    times = {detector_name:[] for detector_name in detectors}

    
    for file in tqdm(list(datasets_info['images'].keys())[::step]):
        file_path = datasets_info['images'][file]['path']
        img = cv2.imread(file_path)
        image_count+=1

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

        for detector_name, detector in detectors.items():
            current_times = []
            for _ in range(NUM_REPEATS):
                boxes, classes, confidences = detector.detect(img)
                current_times.append(detector.get_timing())
            times[detector_name].append(min(current_times))

    results = {}
    results['num_images'] = image_count
    results['mean_times'] = {detector_name: float(np.mean(times[detector_name])) for detector_name in times}
    results['total_area'] = total_area
    results['mean_area'] = total_area/image_count

    with open(output_path, 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False, sort_keys=False)
