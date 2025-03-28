"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ecb import ECB
import torch
import onnx


class FastSCNN_pico_ecb(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN_pico_ecb, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample_ECB(8, 16, 32)
        num_ch = 32
        self.global_feature_extractor = GlobalFeatureExtractor(32, [32, num_ch], 2, [1, 2])
        self.feature_fusion = FeatureFusionModule(32, num_ch, num_ch)
        self.classifier = Classifer(num_ch, num_classes*16)
        self.pixel_shuffle = nn.PixelShuffle(4)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes*16, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = self.pixel_shuffle(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs = x
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = self.pixel_shuffle(auxout)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs = x, auxout
        return outputs
    
    def eval(self):
        for module in self.modules():
            if isinstance(module, ECB):
                module.rep_params()  # Explicitly merge weights
        super().eval()


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
    
    
class _ConvReLU_ECB(nn.Module):
    """Conv-BN-ReLU"""
    def __init__(self, in_channels, out_channels, stride=1, padding=0, **kwargs):
        super(_ConvReLU_ECB, self).__init__()
        self.conv = nn.Sequential(
            ECB(in_channels, out_channels, 2, stride, act_type='linear'),
            #nn.BatchNorm2d(out_channels),
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


class LearningToDownsample_ECB(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=8, dw_channels2=16, out_channels=32, **kwargs):
        super(LearningToDownsample_ECB, self).__init__()
        self.conv1 = _ConvReLU_ECB(1, dw_channels1, 2, 1)
        self.conv2 = _ConvReLU_ECB(dw_channels1, dw_channels2, 2, 1)
        self.conv3 = _ConvReLU_ECB(dw_channels2, out_channels, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
    def eval(self):
        for module in self.modules():
            if isinstance(module, ECB):
                module.rep_params()  # Explicitly merge weights
        super().eval()

class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                  t=6, num_blocks=(3, 3)):
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

if __name__ == '__main__':
    model = FastSCNN_pico_ecb(2, False)
    x = torch.randn(1, 1, 320, 320)
    train_res = model(x)

    # Switch to inference mode
    model.eval()
    eval_res = model(x)

    # Print ECB after calling eval()
    print("Diff:", torch.abs(train_res-eval_res).mean())  # ECB should

    # Export to ONNX
    torch.onnx.export(model, x, "convbnrelu_ecb.onnx", opset_version=19, verbose=True)
    onnx_model = onnx.load("convbnrelu_ecb.onnx")

    # Print ONNX nodes
    for node in onnx_model.graph.node:
        print(node.op_type, node.name)
