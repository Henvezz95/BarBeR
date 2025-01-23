import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from arch import BaFaLo_1
from fast_scnn import FastSCNN
from fast_scnn_nano import FastSCNN_nano
import sys
sys.path.append('./python')
sys.path.append('./algorithms')
from bafalo_detector import BaFaLo_detector
from utils.dataloader import BarcodeDataset
from loss_functions import DefaultLoss, WeightedLoss, WeightedDiceLoss, BoundaryLoss

import yaml
import sys
import getopt
import numpy as np

def im2col_sliding_strided(A, window_size, strides=(1,1), reshape=True):
    m,n = A.shape
    s0, s1 = A.strides
    nrows = (m-window_size[0])//strides[0]+1
    ncols = (n-window_size[1])//strides[1]+1
    shp = window_size[0],window_size[1],nrows,ncols

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=(s0,s1,s0*strides[0],s1*strides[1]))
    if reshape:
        return np.transpose(out_view, (2,3,0,1)).reshape((-1, window_size[0]*window_size[1]))
    return out_view

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
    bafalo_coarse = BaFaLo_detector('./Saved_Models/fscnn_nano_singlech_320_0.pt', 
                                    th=0.5,
                                    minArea=100, 
                                    device = 'cuda' if cuda else 'cpu',
                                    single_class = True)
    def train_one_epoch(cuda=True):
        running_loss = 0.

        for i, data in tqdm(enumerate(dataloader_train)):
            inputs, labels = data
            optimizer.zero_grad()
            loss = 0
            if cuda:
                #for i in range(len(inputs)):
                    #img = inputs[i,0]*255
                    #boxes, _, _ = bafalo_coarse.detect(img[::2,::2,np.newaxis])
                    #for box in boxes:
                    #    x,y,w,h = box*2
                    #    x0 = (max(x,0)//32)*32
                    #    x1 = (min(x+w+32, 640)//32)*32
                    #    y0 = (max(y, 0)//32)*32
                    #    y1 = (min(y+h+32, 640)//32)*32
                    #    if (x1-x0)*(y1-y0) > 200:
                    #        outputs = model(inputs[i:i+1,:,y0:y1,x0:x1].cuda())
                    #        loss += loss_fc(outputs, labels[i:i+1,0:1,y0:y1,x0:x1].cuda())
                outputs = model(inputs.cuda())
                loss = loss_fc(outputs, labels.cuda())
                loss.backward()  # Compute gradients for this image
            else:
                for i in range(len(inputs)):
                    img = inputs[i,0]*255
                    boxes, _, _ = bafalo_coarse.detect(img[::2,::2,np.newaxis])
                    #for box in boxes:
                        #x,y,w,h = box*2
                        #x0 = (max(x,0)//32)*32
                        #x1 = (min(x+w+32, 640)//32)*32
                        #y0 = (max(y, 0)//32)*32
                        #y1 = (min(y+h+32, 640)//32)*32
                        #if (x1-x0)*(y1-y0) > 200:
                        #    outputs = model(inputs[i:i+1,:,y0:y1,x0:x1])
                        #    loss += loss_fc(outputs, labels[i:i+1,0:1,y0:y1,x0:x1])
                    loss += loss_fc(outputs, labels[i:i+1])
            optimizer.step()  # Apply accumulated gradients
            optimizer.zero_grad()  # Clear gradients for the next batch
            running_loss += loss.item()
        return running_loss / (i + 1)
    
    config_path, output_path = parse_inputs(sys.argv[0], sys.argv[1:])
    # Read YAML file
    with open(config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    
    use_ram = data_loaded['use_ram']
    annotations_path = data_loaded['coco_annotations_path']
    images_path = data_loaded['images_path']
    longest_edge_size = data_loaded['imgsz']
    batch_size = data_loaded['batch']
    patience = data_loaded['patience']
    
    train_dataset = BarcodeDataset(f'{annotations_path}train.json', images_path, clahe=False, dist_map=False,
                               max_size=longest_edge_size, downscale_factor=1, remove_first_ch = True, gray_scale = True)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = BarcodeDataset(f'{annotations_path}val.json', images_path, clahe=False, dist_map=False,
                               max_size=longest_edge_size, downscale_factor=1, remove_first_ch = True, gray_scale = True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if cuda:
        #model = BaFaLo_1(n_feats = 16, num_blocks = 3, down_ch = 8, num_classes=1).cuda()
        #model = FastSCNN(num_classes=2).cuda()
        model = FastSCNN_nano(num_classes=2).cuda()
    else:
        model = BaFaLo_1(n_feats = 16, num_blocks = 4, down_ch = 8)
    if data_loaded['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
        
    epoch_number = 0

    loss_fc = DefaultLoss(gamma=3)
    vloss_history = []

    EPOCHS = data_loaded['epochs']
    best_vloss = 1e6

    if use_ram:
        print('The entire dataset will be loaded into RAM. This should significantly speed up the program after the first epoch')

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        print('Learning Rate=', optimizer.param_groups[0]['lr'])

        model.train(True)
        avg_loss = train_one_epoch(cuda)
        scheduler.step()


        running_vloss = 0.0
        model.eval()

        # Disable gradient computation during validation
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(dataloader_val)):
                vinputs, vlabels = vdata
                if cuda:
                    voutputs = model(vinputs.cuda())
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