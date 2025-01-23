import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import sys
sys.path.append('./python/')

from utils.dataloader import BarcodeDataset
from loss_functions import ZharkovLoss, DefaultLoss

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
    cuda = False
    def train_one_epoch(cuda=True):
        running_loss = 0.

        for i, data in tqdm(enumerate(dataloader_train)):
            inputs, labels = data

            optimizer.zero_grad()
            if cuda:
                outputs = model(inputs.cuda())
            else:
                outputs = model(inputs)

            if cuda:
                loss = loss_fc(outputs, labels.cuda())
            else:
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
    
    train_dataset = BarcodeDataset(f'{annotations_path}train.json', images_path, 
                               max_size=longest_edge_size, use_ram=use_ram)
    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_dataset = BarcodeDataset(f'{annotations_path}val.json', images_path, 
                               max_size=longest_edge_size, use_ram=use_ram)
    dataloader_val = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

    if cuda:
        model = models.ZharkovDilatedNet(in_channels=3, num_classes=3).cuda()
    else:
        model = models.ZharkovDilatedNet(in_channels=3, num_classes=3)
    if data_loaded['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    if data_loaded['loss']:
        loss_fc = ZharkovLoss()
    else:
        loss_fc = DefaultLoss()

    epoch_number = 0
    loss_fc =ZharkovLoss()
    vloss_history = []

    EPOCHS = data_loaded['epochs']
    best_vloss = 1e6

    if use_ram:
        print('The entire dataset will be loaded into RAM. This should significantly speed up the program after the first epoch')

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(cuda)


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
            if best_vloss < min(vloss_history[-10:])-1e-12:
                print('Early Stopping!')
                break

        epoch_number += 1