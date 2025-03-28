import torch
import numpy as np

from torch import optim
from loss_functions import FocalLoss

from fast_scnn import FastSCNN, FastSCNN_0_5x, FastSCNN_0_25x
from fast_scnn_pico import BaFaLo_SCNN, BaFaLo_SCNN_noshuffle
from bisenetv2 import BiSeNetV2, BiSeNetV2_0_5x, BiSeNetV2_0_25x, BiSeNetV2_0_125x
from contextnet import ContextNet_0_25x, ContextNet_0_5x, ContextNet


def get_optimizer(network_type, model, num_epochs):
    if network_type in ['fastscnn', 'fastscnn_0.5x', 'fastscnn_0.25x']:
        optimizer = optim.SGD(model.parameters(), lr=0.045, momentum=0.9)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lambda iter: (1 - iter / num_epochs) ** 0.9)
    elif network_type in ['bisenet', 'bisenet_0.5x', 'bisenet_0.25x']:
        conv_params = []
        other_params = []
        weight_decay_value = 0.0005
        for name, param in model.named_parameters():
            # Skip params with no gradient
            if not param.requires_grad:
                continue
        
            # Example logic: if it's a conv weight, go to conv_params
            # (You can refine the condition to check shape, name, or something else)
            if "conv" in name and "weight" in name:  
                conv_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.SGD([
            {'params': conv_params, 'weight_decay': weight_decay_value},
            {'params': other_params, 'weight_decay': 0.0}
        ], lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda iter: (1 - iter / num_epochs) ** 0.9)
    elif network_type in ['contextnet', 'contextnet_0.5x', 'contextnet_0.25x']: 
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=0.045,
                                        alpha=0.9,
                                        momentum=0.9,
                                        eps=1.0) 
        power = 0.98
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** power
        )
    else:
        lr_decay = 1/(np.power(100,1/num_epochs))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    return optimizer, scheduler

def get_training_config(network_type, num_epochs,  in_ch=3, n_classes=2, aux=False):
    loss_fc = FocalLoss(gamma=0, alpha=0.5)
    aux_weight = 0.4
    if network_type == 'fastscnn':
        model = FastSCNN(n_classes, in_ch, aux)
    elif network_type == 'fast_scnn_0.5x':
        model = FastSCNN_0_5x(n_classes, in_ch, aux)
    elif network_type == 'fast_scnn_0.25x':
        model = FastSCNN_0_25x(n_classes, in_ch, aux)
    elif network_type == 'bisenet':
        model = BiSeNetV2(n_classes, in_ch, aux)
        aux_weight = 1.0
    elif network_type == 'bisenet_0.5x':
        model = BiSeNetV2_0_5x(n_classes, in_ch, aux)
        aux_weight = 1.0
    elif network_type == 'bisenet_0.25x':
        model = BiSeNetV2_0_25x(n_classes, in_ch, aux)
        aux_weight = 1.0
    elif network_type == 'bisenet_0.125x':
        model = BiSeNetV2_0_125x(n_classes, in_ch, aux)
        aux_weight = 1.0
    elif network_type == 'contextnet':
        model = ContextNet(n_classes, in_ch)
    elif network_type == 'contextnet_0.5x':
        model = ContextNet_0_5x(n_classes, in_ch)
    elif network_type == 'contextnet_0.25x':
        model = ContextNet_0_25x(n_classes, in_ch)
    elif network_type == 'bafalo_noshuffle':
        model = BaFaLo_SCNN_noshuffle(n_classes, in_ch, mid_ch=32, aux=aux)
    else:
        model = BaFaLo_SCNN(n_classes, in_ch, mid_ch=32, aux=aux)
    optimizer, scheduler = get_optimizer(network_type, model, num_epochs)
    return model, loss_fc, optimizer, scheduler, aux_weight