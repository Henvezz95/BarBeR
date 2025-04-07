import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from arch import BaFaLo_1
from fast_scnn import FastSCNN, FastSCNN_0_5x, FastSCNN_0_25x
from fast_scnn_nano import FastSCNN_nano
from fast_scnn_nano_shuffle import FastSCNN_nano_shuffle
from bisenetv2 import BiSeNetV2, BiSeNetV2_0_5x, BiSeNetV2_0_25x
from contextnet import ContextNet_0_25x, ContextNet_0_5x, ContextNet
from training_config_function import get_training_config 
import sys
sys.path.append('./python')
from utils.dataloader import BarcodeDataset
from utils.utility import tensor_resize
from loss_functions import FocalLoss, WeightedLoss, WeightedDiceLoss, BoundaryLoss

import yaml
import sys
import getopt
import numpy as np

def parse_inputs(file_path, argv):
    config_path = None
    output_path = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["cfile=", "ofolder="])
    except getopt.GetoptError:
        _extracted_from_parse_inputs(file_name)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name, '-c <configfile> -o <trained_model_output_path>')
            print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            config_path = arg
        elif opt in ("-o", "--ofolder"):
            output_path = arg

    if config_path is None:
        _extracted_from_parse_inputs(file_name)
    if output_path is None:
        _extracted_from_parse_inputs(file_name)
    return config_path, output_path


# TODO Rename this here and in `parse_inputs`
def _extracted_from_parse_inputs(file_name):
    print(file_name, '-c <configfile> -o <trained_model_output_path>')
    print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically depending on the format')
    sys.exit(2)


if __name__ == "__main__":
    cuda = True
    add_boundary_loss = False
    def train_one_epoch(cuda=True, crops=False):
        running_loss = 0.

        for i, data in tqdm(enumerate(dataloader_train)):
            inputs, labels = data
            res_alpha = np.random.uniform(min_scale, max_scale)/1.5
            h_new = w_new = int(np.round(res_alpha*longest_edge_size/32))*32
            inputs = tensor_resize(inputs, h_new, w_new, mode='bilinear')
            labels = tensor_resize(labels, h_new, w_new, mode='bilinear')
            optimizer.zero_grad()

            if cuda:
                outputs = model(inputs.cuda())
                if isinstance(outputs, (tuple, list)):
                    loss = loss_fc(outputs[0], labels.cuda())
                    for i in range(1, len(outputs)):
                        loss += loss_fc(outputs[i], labels.cuda())*aux_weight
                else:
                    loss = loss_fc(outputs, labels.cuda())
            else:
                outputs = model(inputs)
                loss = loss_fc(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / (i + 1)
    
    config_path, output_path = parse_inputs(sys.argv[0], sys.argv[1:])
    # Read YAML file
    with open(config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    use_ram = data_loaded['use_ram']
    annotations_path = data_loaded['coco_annotations_path']
    images_path = data_loaded['images_path']
    longest_edge_size = int(data_loaded['imgsz']*1.5)
    batch_size = data_loaded['batch']
    patience = data_loaded['patience']
    model_type = data_loaded['model_type']
    num_epochs = data_loaded['epochs']
    gray_scale = data_loaded['gray_scale']
    aux = data_loaded['aux']
    min_scale, max_scale = data_loaded['scale_range']
    in_ch = 1 if gray_scale else 3
    
    train_dataset = BarcodeDataset(f'{annotations_path}train.json', 
                                   images_path, 
                                   max_size=longest_edge_size, 
                                   downscale_factor = 1, 
                                   remove_first_ch = True, 
                                   gray_scale = gray_scale)
    dataloader_train = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=0)
    val_dataset = BarcodeDataset(f'{annotations_path}val.json', 
                                 images_path, 
                                 max_size=longest_edge_size, 
                                 downscale_factor=1, 
                                 remove_first_ch = True, 
                                 gray_scale = gray_scale)
    dataloader_val = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=0)

    model, loss_fc, optimizer, scheduler, aux_weight = get_training_config(model_type, 
                                                               num_epochs,  
                                                               in_ch, 
                                                               n_classes=2, 
                                                               aux=False)
    if cuda:
        model = model.cuda()
        
    epoch_number = 0
    vloss_history = []
    best_vloss = 1e6

    if use_ram:
        print('The entire dataset will be loaded into RAM. This should significantly speed up the program after the first epoch')

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        print('Learning Rate=', optimizer.param_groups[0]['lr'])

        model.train(True)
        avg_loss = train_one_epoch(cuda, crops=False)
        scheduler.step()

        running_vloss = 0.0
        model.eval()

        # Disable gradient computation during validation
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(dataloader_val)):
                vinputs, vlabels = vdata
                res_alpha = np.random.uniform(min_scale, max_scale)/1.5
                h_new = w_new = int(np.round(res_alpha*longest_edge_size/32))*32
                vinputs = tensor_resize(vinputs, h_new, w_new)
                vlabels = tensor_resize(vlabels, h_new, w_new)
                if cuda:
                    voutputs = model(vinputs.cuda())
                    if isinstance(voutputs, (tuple, list)):
                        vloss = loss_fc(voutputs[0], vlabels.cuda())
                        #for i in range(1, len(voutputs)):
                        #    vloss += loss_fc(voutputs[i], vlabels.cuda())*0.2
                    else:
                        vloss = loss_fc(voutputs, vlabels.cuda())
                else:
                    voutputs = model(vinputs)
                    vloss = loss_fc(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        vloss_history.append(avg_vloss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model, output_path+'.pt')
        else:
            if best_vloss < min(vloss_history[-patience:])-1e-12:
                print('Early Stopping!')
                break

        epoch_number += 1
