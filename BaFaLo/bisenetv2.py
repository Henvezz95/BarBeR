import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class DetailBranch(nn.Module):

    def __init__(self, in_ch=1, mid_ch=64):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, 3, stride=2),
            ConvBNReLU(mid_ch, mid_ch, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(mid_ch, mid_ch, 3, stride=2),
            ConvBNReLU(mid_ch, mid_ch, 3, stride=1),
            ConvBNReLU(mid_ch, mid_ch, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(mid_ch, mid_ch*2, 3, stride=2),
            ConvBNReLU(mid_ch*2, mid_ch*2, 3, stride=1),
            ConvBNReLU(mid_ch*2, mid_ch*2, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):

    def __init__(self, in_ch=1, mid_ch=16):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(mid_ch, mid_ch//2, 1, stride=1, padding=0),
            ConvBNReLU(mid_ch//2, mid_ch, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(mid_ch*2, mid_ch, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):
    def __init__(self, mid_ch=128):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(mid_ch)
        self.conv_gap = ConvBNReLU(mid_ch, mid_ch, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(mid_ch, mid_ch, 3, stride=1)

    def forward(self, x):
        #feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = F.adaptive_avg_pool2d(x, (1, 1))  # ONNX-friendly
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat
            


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self, in_ch=1, ch_width=16):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock(in_ch, mid_ch=ch_width)
        self.S3 = nn.Sequential(
            GELayerS2(ch_width, ch_width*2),
            GELayerS1(ch_width*2, ch_width*2),
        )
        self.S4 = nn.Sequential(
            GELayerS2(ch_width*2, ch_width*4),
            GELayerS1(ch_width*4, ch_width*4),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(ch_width*4, ch_width*8),
            GELayerS1(ch_width*8, ch_width*8),
            GELayerS1(ch_width*8, ch_width*8),
            GELayerS1(ch_width*8, ch_width*8),
        )
        self.S5_5 = CEBlock(mid_ch=ch_width*8)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self, mid_ch=128):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=1,
                padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=1,
                padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)
        return self.conv(left + right)



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(
                mid_chan, n_classes, kernel_size=1, stride=1,
                padding=0, bias=True)

    def forward(self, x, size=None):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        if size is not None:
            feat = F.interpolate(feat, size=size,
                mode='bilinear', align_corners=True)
        return feat


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, in_ch=1, aux=True):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch(in_ch, mid_ch=64)
        self.segment = SegmentBranch(in_ch, ch_width=16)
        self.bga = BGALayer(mid_ch=128)
        self.aux = aux

        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, n_classes)
        self.aux2 = SegmentHead(16, 128, n_classes)
        self.aux3 = SegmentHead(32, 128, n_classes)
        self.aux4 = SegmentHead(64, 128, n_classes)
        self.aux5_4 = SegmentHead(128, 128, n_classes)

        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head, size)
        if self.aux:
            logits_aux2 = self.aux2(feat2, size)
            logits_aux3 = self.aux3(feat3, size)
            logits_aux4 = self.aux4(feat4, size)
            logits_aux5_4 = self.aux5_4(feat5_4, size)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        else:
            return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

class BiSeNetV2_0_5x(BiSeNetV2):
    def __init__(self, n_classes, in_ch=1, aux=True):
        super(BiSeNetV2_0_5x, self).__init__(n_classes, in_ch, aux)
        self.detail = DetailBranch(in_ch, mid_ch=32)
        self.segment = SegmentBranch(in_ch, ch_width=8)
        self.bga = BGALayer(mid_ch=64)
        
        self.head    = SegmentHead(64, 512, n_classes)
        self.aux2    = SegmentHead(8,  64, n_classes)
        self.aux3    = SegmentHead(16, 64, n_classes)
        self.aux4    = SegmentHead(32, 64, n_classes)
        self.aux5_4  = SegmentHead(64, 64, n_classes)
        self.init_weights()

class BiSeNetV2_0_25x(BiSeNetV2):
    def __init__(self, n_classes, in_ch=1, aux=True):
        super(BiSeNetV2_0_25x, self).__init__(n_classes, in_ch, aux)
        self.detail = DetailBranch(in_ch, mid_ch=16)
        self.segment = SegmentBranch(in_ch, ch_width=4)
        self.bga = BGALayer(mid_ch=32)
        
        self.head    = SegmentHead(32, 256, n_classes)
        self.aux2    = SegmentHead(4,  32, n_classes)
        self.aux3    = SegmentHead(8, 32, n_classes)
        self.aux4    = SegmentHead(16, 32, n_classes)
        self.aux5_4  = SegmentHead(32, 32, n_classes)
        self.init_weights()

class BiSeNetV2_0_125x(BiSeNetV2):
    def __init__(self, n_classes, in_ch=1, aux=True):
        super(BiSeNetV2_0_125x, self).__init__(n_classes, in_ch, aux)
        self.detail = DetailBranch(in_ch, mid_ch=8)
        self.segment = SegmentBranch(in_ch, ch_width=2)
        self.bga = BGALayer(mid_ch=16)
        
        self.head    = SegmentHead(16, 128, n_classes)
        self.aux2    = SegmentHead(2,  16, n_classes)
        self.aux3    = SegmentHead(4, 16, n_classes)
        self.aux4    = SegmentHead(8, 26, n_classes)
        self.aux5_4  = SegmentHead(16, 16, n_classes)
        self.init_weights()