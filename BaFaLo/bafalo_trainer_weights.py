import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from arch import BaFaLo_1
from fast_scnn import FastSCNN
from fast_scnn_nano import FastSCNN_nano
from bisenetv2 import BiSeNetV2
from contextnet import ContextNet
import sys
sys.path.append('./python')
from utils.dataloader import BarcodeDataset_with_weights
from loss_functions import DefaultLoss_weighted, WeightedLoss, WeightedDiceLoss, BoundaryLoss

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

def make_crops_from_image(img, num_H, num_W, offsetX, offsetY):
    H,W = img.shape
    x_size = (W-offsetX)//num_W 
    y_size = (H-offsetY)//num_H 
    cropped_image = img[offsetY:offsetY+num_H*y_size, offsetX:offsetX+num_W*x_size]
    img_crops = im2col_sliding_strided(cropped_image, (y_size,x_size), strides=(y_size,x_size), reshape=False)
    img_crops = img_crops.reshape(y_size,x_size,num_H*num_W)
    img_crops = np.transpose(img_crops,(2,0,1))
    return img_crops

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
    def train_one_epoch(cuda=True):
        running_loss = 0.

        for i, data in tqdm(enumerate(dataloader_train)):
            inputs, labels = data
            optimizer.zero_grad()

            if cuda:
                outputs = model(inputs.cuda())
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
    longest_edge_size = data_loaded['imgsz']
    batch_size = data_loaded['batch']
    patience = data_loaded['patience']
    
    train_dataset = BarcodeDataset_with_weights(f'{annotations_path}train.json', images_path, clahe=False,
                               max_size=longest_edge_size, downscale_factor=1, gray_scale = True)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = BarcodeDataset_with_weights(f'{annotations_path}val.json', images_path, clahe=False, 
                               max_size=longest_edge_size, downscale_factor=1, gray_scale = True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if cuda:
        #model = BaFaLo_1(n_feats = 16, num_blocks = 3, down_ch = 8, num_classes=1).cuda()
        #model = BiSeNetV2(n_classes=2, in_ch=1, aux=False).cuda()
        #model = FastSCNN(num_classes=2).cuda()
        #model = FastSCNN_nano(num_classes=2).cuda()
        model = ContextNet(num_class=2, n_channel=1).cuda()
    else:
        model = BaFaLo_1(n_feats = 16, num_blocks = 4, down_ch = 8)
    if data_loaded['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
        
    epoch_number = 0

    #boundary_loss = BoundaryLoss(scale=1e-3)
    loss_fc = DefaultLoss_weighted(gamma=2)
    #loss_fc = WeightedDiceLoss(w1=0.5,w2=0.5)
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