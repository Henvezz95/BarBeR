import numpy as np
import cv2
import json
import sys
import getopt
from tqdm import tqdm

from utils.utility import from_np, intersect_area
from bounding_box import BoundingBox
from evaluators import coco_evaluator
from utils.enumerators import BBType
import yaml
import torch
import os

sys.path.append('./algorithms/detectors/') 
sys.path.append('./algorithms/readers/') 

from zbar_reader import ZbarReader, ZbarReaderWithSegmentation

def reading_rate(decoded, not_decoded):
    return decoded / (not_decoded + decoded)

def parse_inputs(file_path, argv):
    def _parse_inputs_aux(arg0, file_name):
        print(arg0)
        print(file_name, '-c <configfile> -o <outputfolder>')
        sys.exit(2)
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

    if config_path is None:
        _parse_inputs_aux(
            'A configuration file in yaml format is needed to run the program',
            file_name,
        )
    if output_path is None:
        _parse_inputs_aux(
            'Provide a path to save the generated Results', file_name
        )
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
    NUM_REPEATS = 1

    if num_threads > 0:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

    with open(f'{coco_annotation_path}test.json') as json_file:
        coco_annotations = json.load(json_file)

    with open(f'{coco_annotation_path}datasets_info.json') as json_file:
        datasets_info = json.load(json_file)

    detectors = {}
    readers = {'None':ZbarReader(pre_localizer=None)}
    results = {'None':[]}
    times = {'None': []} | {alg['name']: [] for alg in test_config['algorithms']}
    
    for algorithm in test_config['algorithms']:
        mod = import_module(algorithm['library'])
        mod = getattr(mod, algorithm['class'])
        detectors[algorithm['name']] = mod(**algorithm['preloc_args'])
        if algorithm['loc_type'] == 'detect':
            readers[algorithm['name']] = ZbarReader(pre_localizer=detectors[algorithm['name']], 
                                                    **algorithm['reader_args'])
        elif algorithm['loc_type'] == 'segment':
            readers[algorithm['name']] = ZbarReaderWithSegmentation(pre_localizer=detectors[algorithm['name']],
                                                                    **algorithm['reader_args'])
        results[algorithm['name']] = []

    ann_index = 0
    image_counter = 0
    GT_area = 0
    total_area = 0

    num_labels = {'small':0, 'medium':0, 'large':0}
    all_ppe_values = []
    all_areas = []
    datasets_map = {
        dataset: {
            k
            for k in datasets_info['images']
            if datasets_info['images'][k]['dataset'] == dataset
        }
        for dataset in datasets_info['datasets']
    }


    for file_name, metadata in tqdm(datasets_info['images'].items()):
        if datasets_info['images'][file_name]['split'] != 'test':
            continue
        img_path = datasets_info['images'][file_name]['path']
        true_boxes = [np.int32(a) for a in datasets_info['images'][file_name]['boxes'].values()]
        true_types = list(datasets_info['images'][file_name]['types'].values())
        true_classes = [2 if x in ['QR', 'DATAMATRIX', 'AZTEC', 'PDF417', '2D'] else 1 for x in true_types]
        true_ppe = list(datasets_info['images'][file_name]['ppes'].values())
        true_strings = list(datasets_info['images'][file_name]['strings'].values())

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

        total_area += (W_new*H_new)/1e6
        for i in range(len(true_boxes)):
            true_boxes[i][::2] = np.int32(np.round(W_new*true_boxes[-1][::2]/W))
            true_boxes[i][1::2] = np.int32(np.round(H_new*true_boxes[-1][1::2]/H))
            true_ppe[i] = true_ppe[i]/W*W_new

        dataset_name = datasets_info['images'][file_name]['dataset']
        for reader_name, reader in readers.items():
            GT = []
            for i in range(len(true_boxes)):
                area = true_boxes[i][2]*true_boxes[i][3]
                if area < 32**2:
                    area_type = 'small'
                elif area < 96**2:
                    area_type = 'medium'
                else:
                    area_type = 'large'
                GT.append({'box':true_boxes[i], 
                           'string':true_strings[i], 
                           'ppe': true_ppe[i],
                           'type': true_types[i],
                           'class': true_classes[i],
                           'area_type': area_type,
                           'decoded':False})
            current_times = []
            for _ in range(NUM_REPEATS):
                decoded_labels = reader.decode(img)
                current_times.append(reader.get_timing())
            times[reader_name].append(min(current_times))
            for dec_label in decoded_labels:
                for item in GT:
                    area_of_intersection = intersect_area(dec_label['box'], item['box'])
                    label_area = dec_label['box'][2]*dec_label['box'][3]
                    if (
                        area_of_intersection / (label_area+1e-3) > 0.5
                        and item['string'] == dec_label['string']
                    ):
                        item['decoded'] = True
                        break
                    if (
                        area_of_intersection / (label_area+1e-3) > 0.5
                        and item['string'] == '-1'
                    ):
                        item['decoded'] = True
                        break
            results[reader_name].extend(GT)

    filtered_results = {
        reader_name: {
            'mean_time': float(np.mean(times[reader_name])),
            'decoded_validated': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and not x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'decoded_validated_1D': len(
                list(
                    filter(
                        lambda x: x['type'] in ['EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated_1D': len(
                list(
                    filter(
                        lambda x: x['type'] in ['EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and not x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'decoded_validated_2D': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR']
                          and x['string'] != '-1' 
                          and x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated_2D': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR']
                          and x['string'] != '-1' 
                          and not x['decoded'],
                        results[reader_name],
                    )
                )
            ),
            'decoded_validated_small': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and x['decoded']  
                          and x['area_type']=='small',
                        results[reader_name],
                    )
                )
            ),
            'decoded_validated_medium': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and x['decoded']  
                          and x['area_type']=='medium',
                        results[reader_name],
                    )
                )
            ),
            'decoded_validated_large': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and x['decoded']  
                          and x['area_type']=='large',
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated_small': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and not x['decoded']  
                          and x['area_type']=='small',
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated_medium': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and not x['decoded']  
                          and x['area_type']=='medium',
                        results[reader_name],
                    )
                )
            ),
            'not_decoded_validated_large': len(
                list(
                    filter(
                        lambda x: x['type'] in ['QR', 'EAN13','I2O5','C39','EAN8','C128']
                          and x['string'] != '-1' 
                          and not x['decoded']  
                          and x['area_type']=='large',
                        results[reader_name],
                    )
                )
            ),
            'decoded': len(
                list(filter(lambda x: x['decoded'], results[reader_name]))
            ),
            'not_decoded': len(
                list(
                    filter(
                        lambda x: not x['decoded'], results[reader_name]
                    )
                )
            ),
        }
        for reader_name in results
    }

    for filt_result in filtered_results.values():
        filt_result['reading_rate_validated'] = reading_rate(
            filt_result['decoded_validated'], filt_result['not_decoded_validated']
        )
        filt_result['reading_rate_validated_1D'] = reading_rate(
            filt_result['decoded_validated_1D'], filt_result['not_decoded_validated_1D']
        )
        filt_result['reading_rate_validated_2D'] = reading_rate(
            filt_result['decoded_validated_2D'], filt_result['not_decoded_validated_2D']
        )
        filt_result['reading_rate'] = reading_rate(
            filt_result['decoded'], filt_result['not_decoded']
        )
    with open(output_path, 'w') as outfile:
        yaml.dump(filtered_results, outfile, default_flow_style=False, sort_keys=False)
