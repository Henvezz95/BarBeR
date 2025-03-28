###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
###########################################################################

"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, in_ch=1, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64, in_ch=in_ch)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs = x
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs = x, auxout
        return outputs
    
class FastSCNN_0_5x(nn.Module):
    def __init__(self, num_classes, in_ch=1, aux=False, **kwargs):
        super(FastSCNN_0_5x, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(16, 24, 32, in_ch=in_ch)
        self.global_feature_extractor = GlobalFeatureExtractor(32, [32, 48, 64], 64, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(32, 64, 64)
        self.classifier = Classifer(64, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(16, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs = x
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs = x, auxout
        return outputs
    
class FastSCNN_0_25x(nn.Module):
    def __init__(self, num_classes, in_ch=1, aux=False, **kwargs):
        super(FastSCNN_0_25x, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(8, 12, 16, in_ch=in_ch)
        self.global_feature_extractor = GlobalFeatureExtractor(16, [16, 24, 32], 32, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(16, 32, 32)
        self.classifier = Classifer(32, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(8, 8, 3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(8, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs = x
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs = x, auxout
        return outputs

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class MyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        """
        Mimics nn.AdaptiveAvgPool2d by computing, for each output cell,
        the pooling window as:
            start = floor(i * input_size / output_size)
            end   = ceil((i+1) * input_size / output_size)
        and then averaging over that region.
        """
        super(MyAdaptiveAvgPool2d, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.out_size = output_size

    def forward(self, x):
        N, C, H, W = x.shape
        out_h, out_w = self.out_size
        # Prepare output tensor.
        out = x.new_empty((N, C, out_h, out_w))
        # Loop over each output index.
        for i in range(out_h):
            # Compute vertical boundaries for output row i.
            start_h = math.floor(i * H / out_h)
            end_h = math.ceil((i + 1) * H / out_h)
            for j in range(out_w):
                # Compute horizontal boundaries for output column j.
                start_w = math.floor(j * W / out_w)
                end_w = math.ceil((j + 1) * W / out_w)
                # Slice the input region and compute its mean.
                region = x[:, :, start_h:end_h, start_w:end_w]
                out[:, :, i, j] = region.mean(dim=(-1, -2))
        return out
     
class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        #avgpool = nn.AdaptiveAvgPool2d(size)
        avgpool = MyAdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, in_ch=1):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(in_ch, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x
    

    
def test_adaptive_pool_equivalence():
    torch.manual_seed(0)
    input_tensor = torch.randn(1, 3, 32, 32)  # example input
    out_size = (6, 6)

    # Built-in adaptive avg pooling
    builtin_pool = nn.AdaptiveAvgPool2d(out_size)
    output_builtin = builtin_pool(input_tensor)

    # Custom pooling
    custom_pool = MyAdaptiveAvgPool2d(out_size)
    output_custom = custom_pool(input_tensor)

    print("Output from nn.AdaptiveAvgPool2d:")
    print(output_builtin)
    print("\nOutput from MyAdaptiveAvgPool2d:")
    print(output_custom)
    print("\nAre the outputs close? ", torch.allclose(output_builtin, output_custom, atol=1e-5))

# A simple model that uses our custom pooling so we can export it to ONNX.
class ModelWithCustomPool(nn.Module):
    def __init__(self, output_size):
        super(ModelWithCustomPool, self).__init__()
        self.pool = MyAdaptiveAvgPool2d(output_size)
    def forward(self, x):
        return self.pool(x)

def export_to_onnx():
    # Use an input size that is not necessarily a multiple of the output size.
    input_tensor = torch.randn(1, 3, 1, 4)
    model = ModelWithCustomPool((5, 6))
    # Export with a supported opset version (e.g., 17)
    torch.onnx.export(model, input_tensor, "my_adaptive_avg_pool.onnx", opset_version=17)
    print("\nExported ModelWithCustomPool to ONNX successfully.")

if __name__ == '__main__':
    test_adaptive_pool_equivalence()
    export_to_onnx()