import torch
from torch import nn
import torchvision
import sys
sys.path.append('./python')

from typing import List
from torch import Tensor, einsum

class DefaultLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(DefaultLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions, ground_truth):
        # Detection Loss
        return torchvision.ops.sigmoid_focal_loss(predictions, ground_truth, alpha = self.alpha, gamma = self.gamma, reduction='mean')
    
class WeightedLoss(nn.Module):
    def __init__(self, w1=10, w2=50, gamma=2.0, alpha=0.25):
        super(WeightedLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.gamma = gamma
        self.alpha =alpha
    
    def forward(self, predictions, ground_truth):
        # Detection Loss
        l1 = self.w1*torchvision.ops.sigmoid_focal_loss(predictions[:,0], ground_truth[:,0], alpha = self.alpha, gamma = self.gamma, reduction='mean')
        l2 = self.w2*torchvision.ops.sigmoid_focal_loss(predictions[:,1], ground_truth[:,1], alpha = self.alpha, gamma = self.gamma, reduction='mean')
        return l1+l2
    
class WeightedDiceLoss(nn.Module):
    def __init__(self, w1=0.5, w2=0.5):
        super(WeightedDiceLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
    
    def forward(self, predictions, ground_truth):
        # Detection Loss
        predictions = nn.Sigmoid()(predictions)
        numerator_1 = 2*torch.sum(predictions[:,0]*ground_truth[:,0])+1e-3
        denominator_1 = torch.sum(predictions[:,0])+torch.sum(ground_truth[:,0])+2e-3
        numerator_2 = 2*torch.sum(predictions[:,1]*ground_truth[:,1])+1e-3
        denominator_2 = torch.sum(predictions[:,1])+torch.sum(ground_truth[:,1])+2e-3
        l1 = 1 - numerator_1/denominator_1       
        l2 = 1 - numerator_2/denominator_2
        return self.w1*l1+self.w2*l2
    
class BoundaryLoss():
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        multipled =  nn.Sigmoid()(probs)*dist_maps
        return multipled.mean()*self.scale