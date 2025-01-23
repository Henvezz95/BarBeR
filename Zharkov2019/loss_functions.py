import torch
from torch import nn

class ZharkovLoss(nn.Module):
    def __init__(self):
        self.wp=15, 
        self.wn=1, 
        self.wh=5, 
        self.alpha=1
        super(ZharkovLoss, self).__init__()

    def forward(self, predictions, ground_truth):
        # Detection Loss
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = BCE(predictions[:,0,:,:], ground_truth[:,0,:,:])

        Lp = (bce_loss * ground_truth[:,0,:,:].float()).sum()
        non_zero_elements = ground_truth[:,0,:,:].sum()+1e-3
        Lp =  (Lp / non_zero_elements)
        Lp = 15*Lp
        negative_mask = (1-ground_truth[:,0,:,:].float())+1e-3
        Ln = (bce_loss * negative_mask).sum()
        zero_elements = negative_mask.sum()
        Ln = Ln / zero_elements
        Ln = 1 * Ln
        Lh = 0
        for i in range(len(ground_truth)):
            k = ground_truth[i,0,:,:].sum().int()
            if k > 0:
                values, _ = torch.topk((predictions[i,0,:,:]*negative_mask[i]).ravel(), k, largest=True, sorted=True)
                Lh += nn.BCEWithLogitsLoss(reduction='mean')(values, torch.zeros_like(values))
        Lh = Lh / len(ground_truth)
        Lh = 5 * Lh

        # Classification Loss
        Lc = nn.functional.cross_entropy(predictions[:,1:,:,:], ground_truth[:,1:,:,:], reduction='none')
        Lc = Lc*ground_truth[:,0,:,:]
        Lc = Lc.sum()/non_zero_elements
        return Lp+Ln+Lh+Lc

class DefaultLoss(nn.Module):
    def __init__(self):
        self.wp=15, 
        self.wn=1, 
        self.wh=5, 
        self.alpha=1
        super(DefaultLoss, self).__init__()
    
    def forward(self, predictions, ground_truth):
        # Detection Loss
        BCE = nn.BCEWithLogitsLoss(reduction='mean')
        Ld = BCE(predictions[:,0,:,:], ground_truth[:,0,:,:])
        # Classification Loss
        Lc = nn.functional.cross_entropy(predictions[:,1:,:,:], ground_truth[:,1:,:,:], reduction='mean')
        return Ld+Lc