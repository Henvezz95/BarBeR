import os
import torch
import shutil
from ultralytics import YOLO, RTDETR
import shutil
import yaml
import sys
import getopt

def parse_inputs(file_path, argv):
    config_path = None
    output_path = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["cfile=", "ofolder="])
    except getopt.GetoptError:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name, '-c <configfile> -o <trained_model_output_path>')
            print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            config_path = arg
        elif opt in ("-o", "--ofolder"):
            output_path = arg

    if config_path == None:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
        sys.exit(2)
    if output_path == None:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
        sys.exit(2)

    return config_path, output_path



# sourcery skip: avoid-builtin-shadow
if __name__ == "__main__":
    config_path, output_path = parse_inputs(sys.argv[0], sys.argv[1:])

    # Read YAML file
    with open(config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    dataset_path = data_loaded["dataset_path"]
    if dataset_path[0] == '.':
        dataset_path = os.getcwd()+dataset_path[1:]

    torch.cuda.is_available()
    device = torch.device('cuda:0')

    if data_loaded['model_type'] in ['yolo', 'YOLO', 'Yolo']:
        model = YOLO(data_loaded["pretrained_model"])  # load a pretrained model (recommended for training)
    elif data_loaded['model_type'] in ['rtdetr', 'RTDETR', 'Rtdetr']:
        model = RTDETR(data_loaded["pretrained_model"])

    model.to(device)
    result = model.train(data=f'{dataset_path}data.yaml', **data_loaded['args'])

    format = data_loaded['format']
    if format == 'pytorch':
        model.export()
    else:
        model.export(format=format)

    if format == 'onnx':
        extension = '.onnx'
    elif format == 'pytorch':
        extension = '.pt'
    elif format == 'torchscript':
        extension = '.torchscript'
    elif format in ['tflite', 'edgetpu']:
        extension = '.tflite'
    elif format == 'engine':
        extension = '.engine'
    elif format == 'pb':
        extension = '.pb'
    else:
        extension = '.pt'

    shutil.move(f'{str(result.save_dir)}/weights/best{extension}', 
                f'{output_path}{extension}')