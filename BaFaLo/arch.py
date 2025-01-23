"""
All custom models used
"""
from typing import List, Dict, Any

from torch import nn

class ResBlock(nn.Module):
    def __init__(self, n_feats, dil1 = 1, dil2 = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding='same', dilation = dil1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding='same', dilation = dil2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + x
        return  out
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, stride = 1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class BaFaLo_module(nn.Module):
    def __init__(self, n_feats = 16, num_blocks = 3, down_ch = 8, num_classes=2):
        super().__init__()
        self.num_blocks = num_blocks
        self.strided_conv1 = nn.Conv2d(1, down_ch, kernel_size=5, stride=2, padding=2)
        self.strided_conv2 = nn.Conv2d(down_ch, n_feats, kernel_size=3, stride=2, padding=1)
        self.strided_conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1)
        self.channelwise_conv = nn.Conv2d(n_feats, 16*num_classes, kernel_size=1, padding='same')
        body = []
        for _ in range(self.num_blocks):
            body.extend((ResBlock(n_feats), nn.ReLU(True)))
        self.body = nn.Sequential(*body)

        self.act = nn.ReLU(True)
        self.up = nn.PixelShuffle(4)
    def forward(self, x):
        x = self.strided_conv1(x)
        x = self.act(x)
        x = self.strided_conv2(x)
        x = self.act(x)
        x = self.strided_conv3(x)
        x = self.act(x)
        x = self.body(x)
        x = self.channelwise_conv(x)
        x = self.up(x)
        return x
      
class BaFaLo_1(nn.Module):
    def __init__(self, n_feats = 16, num_blocks = 3, down_ch = 8, num_classes=2):
        super().__init__()
        self.bafalo_module = BaFaLo_module(n_feats, num_blocks, down_ch, num_classes)

    def forward(self, x):
        downscale_factor = 2
        out = self.bafalo_module(x)
        out = nn.Upsample(scale_factor=downscale_factor, mode='bilinear')(out)
        return out

class SepConvBaFaLo(nn.Module):
    def __init__(self, ch_start = 16, c_body = 64, initial_ker_size=3, downscale_more = True, num_blocks = 4):
        super().__init__()
        self.downscale_more = downscale_more
        padding = initial_ker_size//2
        self.strided_conv1 = nn.Conv2d(1, ch_start, kernel_size=initial_ker_size, stride=2, padding=padding)
        self.down2 = nn.PixelUnshuffle(2)
        self.up2 = nn.PixelShuffle(4)
        stride = 2 if downscale_more else 1  
        self.sepconv1 = DepthwiseSeparableConv(ch_start*4, c_body, stride=stride)
        body = []
        for _ in range(num_blocks):
            body.extend((DepthwiseSeparableConv(c_body,c_body), nn.ReLU(True)))
        self.body = nn.Sequential(*body)
        self.channelwise_conv = nn.Conv2d(c_body, 32, kernel_size=1, padding='same')
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.strided_conv1(x)
        x = self.act(x)
        x = self.down2(x)
        x = self.sepconv1(x)
        x = self.act(x)
        x = self.body(x)

        x = self.channelwise_conv(x)
        out = self.up2(x)
        if self.downscale_more:
            out = nn.Upsample(scale_factor=2, mode='bilinear')(out)
        return out
