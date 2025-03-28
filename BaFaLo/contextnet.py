import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type in activation_hub:
            self.activation = activation_hub[act_type](**kwargs)
        else:
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

    def forward(self, x):
        return self.activation(x)
    
# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)

# Depth-wise seperable convolution with batchnorm and activation
class DSConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        super().__init__(
            DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, **kwargs),
            PWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class PWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu', bias=True, **kwargs):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, (list, tuple)):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )



class ContextNet(nn.Module):
    def __init__(self, num_class=1, n_channel=1, act_type='relu'):
        super().__init__()
        self.full_res_branch = Branch_1(n_channel, [32, 64, 128], 128, act_type=act_type)
        self.lower_res_branch = Branch_4(n_channel, 128, mult=4, act_type=act_type)
        self.feature_fusion = FeatureFusion(128, 128, 128, act_type=act_type)
        self.classifier = ConvBNAct(128, num_class, 1, act_type='none')

    def forward(self, x):
        size = x.size()[2:]
        x_lower = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        full_res_feat = self.full_res_branch(x)
        lower_res_feat = self.lower_res_branch(x_lower)
        x = self.feature_fusion(full_res_feat, lower_res_feat)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x
    
class ContextNet_0_5x(ContextNet):
    def __init__(self, num_class=1, n_channel=1, act_type='relu'):
        super().__init__(num_class, n_channel, act_type)
        self.full_res_branch = Branch_1(n_channel, [16, 32, 64], 64, act_type=act_type)
        self.lower_res_branch = Branch_4(n_channel, 64, mult=2, act_type=act_type)
        self.feature_fusion = FeatureFusion(64, 64, 64, act_type=act_type)
        self.classifier = ConvBNAct(64, num_class, 1, act_type='none')
    
class ContextNet_0_25x(ContextNet):
    def __init__(self, num_class=1, n_channel=1, act_type='relu'):
        super().__init__(num_class, n_channel, act_type)
        self.full_res_branch = Branch_1(n_channel, [8, 16, 32], 32, act_type=act_type)
        self.lower_res_branch = Branch_4(n_channel, 32, mult=1, act_type=act_type)
        self.feature_fusion = FeatureFusion(32, 32, 32, act_type=act_type)
        self.classifier = ConvBNAct(32, num_class, 1, act_type='none')


class Branch_1(nn.Sequential):
    def __init__(self, in_channels, hid_channels, out_channels, act_type='relu'):
        assert len(hid_channels) == 3
        super().__init__(
                ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
                DWConvBNAct(hid_channels[0], hid_channels[0], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[0], hid_channels[1], act_type=act_type),
                DWConvBNAct(hid_channels[1], hid_channels[1], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[1], hid_channels[2], act_type=act_type),
                DWConvBNAct(hid_channels[2], hid_channels[2], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[2], out_channels, act_type=act_type)
        )


class Branch_4(nn.Module):
    def __init__(self, in_channels, out_channels, mult=4, act_type='relu'):
        super().__init__()
        self.conv_init = ConvBNAct(in_channels, 8*mult, 3, 2, act_type=act_type)
        inverted_residual_setting = [
                # t, c, n, s
                [1, 8*mult, 1, 1],
                [6, 8*mult, 1, 1],
                [6, 12*mult, 3, 2],
                [6, 16*mult, 3, 2],
                [6, 24*mult, 2, 1],
                [6, 32*mult, 2, 1],
            ]

        features = []
        in_channels = 8*mult
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
        self.bottlenecks = nn.Sequential(*features)
        self.conv_last = ConvBNAct(32*mult, out_channels, 3, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, branch_1_channels, branch_4_channels, out_channels, act_type='relu'):
        super().__init__()
        self.branch_1_conv = conv1x1(branch_1_channels, out_channels)
        self.branch_4_conv = nn.Sequential(
                                DSConvBNAct(branch_4_channels, out_channels, 3, dilation=4, act_type='none'),
                                conv1x1(out_channels, out_channels)
                                )
        self.act = Activation(act_type=act_type)                                 

    def forward(self, branch_1_feat, branch_4_feat):
        size = branch_1_feat.size()[2:]

        branch_1_feat = self.branch_1_conv(branch_1_feat)

        branch_4_feat = F.interpolate(branch_4_feat, size, mode='bilinear', align_corners=True)
        branch_4_feat = self.branch_4_conv(branch_4_feat)

        res = branch_1_feat + branch_4_feat
        res = self.act(res)

        return res


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6, act_type='relu'):
        super().__init__()
        hid_channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
                        PWConvBNAct(in_channels, hid_channels, act_type=act_type),
                        DWConvBNAct(hid_channels, hid_channels, 3, stride, act_type=act_type),
                        ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                    )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)