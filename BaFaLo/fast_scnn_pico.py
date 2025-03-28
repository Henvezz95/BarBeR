"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class BaFaLo_SCNN(nn.Module):
    def __init__(self, num_classes, in_ch=1, mid_ch=32, aux=False, **kwargs):
        super(BaFaLo_SCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(in_ch, 12, 24, 32)
        self.global_feature_extractor = GlobalFeatureExtractor(32, [32, mid_ch], mid_ch, 2, [1, 2])
        self.feature_fusion = FeatureFusionModule(32, mid_ch, mid_ch)
        self.classifier = Classifer(mid_ch, num_classes*64)
        self.pixel_shuffle = nn.PixelShuffle(8)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes*64, 1)
            )

    def forward(self, x):
        #size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = self.pixel_shuffle(x)
        #x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs = x
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = self.pixel_shuffle(auxout)
            #auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs = x, auxout
        return outputs
    
class BaFaLo_SCNN_noshuffle(nn.Module):
    def __init__(self, num_classes, in_ch=1, mid_ch=32, aux=False, **kwargs):
        super(BaFaLo_SCNN_noshuffle, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(in_ch, 12, 24, 32)
        self.global_feature_extractor = GlobalFeatureExtractor(32, [32, mid_ch], mid_ch, 2, [1, 2])
        self.feature_fusion = FeatureFusionModule(32, mid_ch, mid_ch)
        self.classifier = Classifer(mid_ch, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes*64, 1)
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


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, in_ch=1, dw_channels1=8, dw_channels2=16, out_channels=32, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv1 = _ConvBNReLU(in_ch, dw_channels1, 3, 2, 1)
        self.conv2 = _ConvBNReLU(dw_channels1, dw_channels2, 3, 2, 1)
        self.conv3 = _ConvBNReLU(dw_channels2, out_channels, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = [block(inplanes, planes, t, stride)]
        layers.extend(block(planes, planes, t, 1) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = nn.Conv2d(lower_in_channels, out_channels, 3, padding=1)
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
        lower_res_feature = F.interpolate(lower_res_feature, 
                                          scale_factor=4, 
                                          mode='bilinear', 
                                          align_corners=True)
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