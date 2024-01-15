import numpy as np
import json
from tqdm import tqdm
import getopt, os
import sys

def parse_inputs(file_path, argv):
    annotations = None
    output_path = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["cfile=", "ofolder="])
    except getopt.GetoptError:
        print(file_name,  '-c <coco_annotations_folder> -o <output_folder>')
        print('The configuration file must be in json format')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name,  '-c <coco_annotations_folder> -o <output_folder>')
            print('The configuration file must be in json format')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            annotations = arg
        elif opt in ("-o", "--ofolder"):
            output_path = arg

    if annotations == None:
        print('Coco annotation folder is needed')
        print(file_name, '-c <coco_annotations_folder> -o <output_folder>')
        sys.exit(2)
    if output_path == None:
        print('Provide a folder to save the generated labels')
        print(file_name, '-c <coco_annotations_folder> -o <output_folder>')
        sys.exit(2)

    if annotations[-1] not in ['/', '\\']:
        annotations+='/'       
    if output_path[-1] not in ['/', '\\']:
        output_path+='/' 

    if output_path[0] == '.':
        output_path = os.getcwd()+output_path[1:]                               
    return annotations, output_path

if __name__ == "__main__":
    annotation_folder, output_path = parse_inputs(sys.argv[0], sys.argv[1:])
    metadata = {}
    annotations = [annotation_folder+file_name for file_name in ['train.json', 'val.json', 'test.json']]
    train_val_test_split = {}

    with open(annotation_folder+'datasets_info.json') as json_file:
        datasets_info = json.load(json_file)

    for annotation_path in annotations:
        with open(annotation_path) as json_file:
            coco_annotations = json.load(json_file)
        annotation_name = os.path.basename(annotation_path)
        train_val_test_split[annotation_name.split('.')[0]] = [datasets_info['images'][x['file_name']]['path']
                                                                for x in coco_annotations['images']]
        
        ann_index = 0
        image_counter = 0
        
        for image_annotation in coco_annotations['images']:
            id = image_annotation['id']
            file_name = image_annotation['file_name']
            H = image_annotation['height']
            W = image_annotation['width']
            true_boxes = []
            metadata[file_name] = []

            while coco_annotations['annotations'][ann_index]['image_id'] == id:
                true_boxes.append(np.array([coco_annotations['annotations'][ann_index]['category_id']-1,
                                            *coco_annotations['annotations'][ann_index]['bbox'], 
                                            ]))
                
                ann_index+=1
                if ann_index >= len(coco_annotations['annotations']):
                    break

            for box in true_boxes:
                normalized_box = [box[0], (box[1]+box[3]/2)/W, (box[2]+box[4]/2)/H, box[3]/W, box[4]/H]
                metadata[file_name].append(normalized_box)

    print(len(metadata)) 
    if not os.path.isdir(output_path+'labels/'):
        os.mkdir(output_path+'labels/')
    for file_name in tqdm(metadata):
        f = open(output_path+'labels/'+file_name[:-3]+'txt', "w")
        f.close()
        f = open(output_path+'labels/'+file_name[:-3]+'txt', "a")
        for box in metadata[file_name]:
            f.write(" ".join([str(x) for x in box])+'\n')
        f.close()

    #Generate train.txt, val.txt, test.txt files
    string = ""

    for img_path in train_val_test_split['train']:
        file_name = os.path.basename(img_path)
        string = f'{string}{output_path}images/{file_name}' + '\n'

    with open(f'{output_path}train.txt', 'w') as f:
        f.write(string)

    string = ""

    for img_path in train_val_test_split['test']:
        file_name = os.path.basename(img_path)
        string = f'{string}{output_path}images/{file_name}' + '\n'

    with open(f'{output_path}test.txt', 'w') as f:
        f.write(string)

    string = ""

    for img_path in train_val_test_split['val']:
        file_name = os.path.basename(img_path)
        string = f'{string}{output_path}images/{file_name}' + '\n'

    with open(f'{output_path}val.txt', 'w') as f:
        f.write(string)


