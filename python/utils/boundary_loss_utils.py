from operator import itemgetter
from functools import partial
from typing import Callable, Tuple, Union, cast, Set, Iterable
from torchvision import transforms
from torch import Tensor
from scipy.ndimage import distance_transform_edt as eucl_distance

import torch
from PIL import Image
import numpy as np
import cv2
import itertools

def calc_dist_map_single(posmask):
    negmask = 1 - posmask
    eucl_dist = lambda x: cv2.distanceTransform(x.astype(np.uint8), cv2.DIST_L2, 5)
    if np.any(posmask):
        return eucl_dist(negmask) * negmask - (eucl_dist(posmask) - 1) * posmask
    return np.zeros_like(posmask, dtype=np.float32)

def calc_dist_map_chw(posmask_chw):
    # Extract dimensions
    C, H, W = posmask_chw.shape
    result = np.zeros_like(posmask_chw, dtype=np.float32)

    for c in range(C):
        posmask = posmask_chw[c, :, :]  # Extract 2D slice
        if np.any(posmask):
            negmask = 1 - posmask
            eucl_dist = lambda x: cv2.distanceTransform(x, cv2.DIST_L2, 0)
            result[c, :, :] = eucl_dist(negmask) * negmask - (eucl_dist(posmask) - 1) * posmask
    return result

def calc_dist_map_bchw(posmask_batch):
    # Extract dimensions
    B, C, H, W = posmask_batch.shape
    result = np.zeros_like(posmask_batch, dtype=np.float32)

    for b, c in itertools.product(range(B), range(C)):
        posmask = posmask_batch[b, c, :, :]  # Extract 2D slice
        if np.any(posmask):
            negmask = 1 - posmask
            eucl_dist = lambda x: cv2.distanceTransform(x, cv2.DIST_L2, 0)
            result[b, c, :, :] = eucl_dist(negmask) * negmask - (eucl_dist(posmask) - 1) * posmask
    return result

'''
D = Union[Image.Image, np.ndarray, Tensor]

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape  # type: ignore # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])

def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),
                lambda t: t.cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])
'''