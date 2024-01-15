import os
import os
import torch
import shutil
from ultralytics import YOLO, RTDETR
import shutil
import yaml

# Read YAML file
with open("./config/ultralytics_training_config.yaml", 'r') as stream:
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
elif format == 'tflite' or format == 'edgetpu':
    extension = '.tflite'
elif format == 'engine':
    extension = '.engine'
elif format == 'pb':
    extension = '.pb'
else:
    extension = '.pt'

shutil.move(f'{str(result.save_dir)}/weights/best{extension}', 
            f'{data_loaded["output_path"]}{extension}')