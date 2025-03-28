import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier, stride=1):
        super(SeqConv3x3, self).__init__()
        
        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.stride = stride  # Will be used on the 3x3 portion

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)

            # 1x1 always stride=1
            conv0 = nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, 
                              padding=0, stride=1)  # CHANGE: fixed stride=1
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # 3x3 uses the user-specified stride
            conv1 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3, 
                              padding=0, stride=self.stride)  # no built-in padding, we do it manually
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, 
                              padding=0, stride=1)  # CHANGE: fixed stride=1
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            self.bias = nn.Parameter(bias)

            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                # Sobel X-like kernel
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, 
                              padding=0, stride=1)  # CHANGE: fixed stride=1
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            self.bias = nn.Parameter(bias)

            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                # Sobel Y-like kernel
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, 
                              padding=0, stride=1)  # CHANGE: fixed stride=1
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            self.bias = nn.Parameter(bias)

            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                # Laplacian kernel
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        """
        For conv1x1-conv3x3:
          - 1x1 has stride=1 (reduce channels only)
          - 3x3 uses self.stride
        For the edge-oriented variants (sobel, laplacian), 
          we do 1x1 (stride=1) then a grouped 3x3 with stride=self.stride.
        """
        if self.type == 'conv1x1-conv3x3':
            # 1x1 with stride=1
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)  # no down-sampling yet

            # manually pad the feature map with bias
            y0 = F.pad(y0, (1, 1, 1, 1), mode='constant', value=0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad

            # final 3x3 with stride=self.stride
            y1 = F.conv2d(y0, self.k1, self.b1, stride=self.stride)
            return y1

        else:
            # (sobel, laplacian, etc.) do 1x1 with stride=1
            y0 = F.conv2d(x, self.k0, self.b0, stride=1)

            # manually pad
            y0 = F.pad(y0, (1, 1, 1, 1), mode='constant', value=0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad

            # grouped 3x3 with stride=self.stride
            y1 = F.conv2d(y0, self.scale * self.mask, self.bias,
                          stride=self.stride, groups=self.out_planes)
            return y1

    def rep_params(self):
        """
        Merges 1x1 and 3x3 into a single 3x3 kernel + bias with stride=self.stride.
        """
        device = self.k0.device  # PyTorch >= 1.5
        if self.type == 'conv1x1-conv3x3':
            # Re-param for 'conv1x1-conv3x3'
            # (1) Weighted kernel = conv(k1) with weight k0
            #     i.e. K1 * K0 in “depth” dimension
            RK = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
            # (2) Weighted bias:
            RB_ones = torch.ones(1, self.mid_planes, 3, 3, device=device) 
            RB_ones = RB_ones * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(RB_ones, self.k1).view(-1,) + self.b1
        else:
            # Re-param for the edge-oriented variants
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            # Now convolve k1 by k0
            RK = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
            RB_ones = torch.ones(1, self.out_planes, 3, 3, device=device)
            RB_ones = RB_ones * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(RB_ones, k1).view(-1,) + self.bias
        return RK, RB


class ECB(nn.Module):
    """
    Edge-oriented Convolution Block that becomes a single stride-S 3x3 in inference.
    """
    def __init__(self, inp_planes, out_planes, depth_multiplier,
                 stride=1, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.stride = stride

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        # The "regular" 3x3 has stride=self.stride
        self.conv3x3 = nn.Conv2d(self.inp_planes, self.out_planes,
                                 kernel_size=3, padding=1, stride=stride)

        # The "1x1 -> 3x3" pipeline
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', 
                                      self.inp_planes,
                                      self.out_planes,
                                      depth_multiplier,
                                      stride=stride)

        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx',
                                      self.inp_planes,
                                      self.out_planes,
                                      -1,
                                      stride=stride)

        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely',
                                      self.inp_planes,
                                      self.out_planes,
                                      -1,
                                      stride=stride)

        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian',
                                      self.inp_planes,
                                      self.out_planes,
                                      -1,
                                      stride=stride)

        # Activation
        if act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif act_type == 'softplus':
            self.act = nn.Softplus()
        elif act_type == 'linear':
            self.act = nn.Identity()
        else:
            raise ValueError('Unsupported activation: ' + act_type)

    def forward(self, x):
        if self.training:
            # Just sum everything (they each do their own stride).
            y = ( self.conv3x3(x)
                  + self.conv1x1_3x3(x)
                  + self.conv1x1_sbx(x)
                  + self.conv1x1_sby(x)
                  + self.conv1x1_lpl(x) )
            if self.with_idt:
                y += x
        else:
            y = self.conv3x3(x)
        return self.act(y)

    def rep_params(self):
        """
        Combine each sub-block’s kernel & bias into one single 3x3 + bias.
        """
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()

        # Summation
        RK = K0 + K1 + K2 + K3 + K4
        RB = B0 + B1 + B2 + B3 + B4

        # If with_idt, that means add identity
        if self.with_idt:
            device = RK.device
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            RK += K_idt
            # no extra bias
        self.conv3x3.weight.data = RK
        self.conv3x3.bias.data = RB

    def eval(self):
        # Automatically converts the ECB block to inference mode.
        self.rep_params()
        super().eval()


if __name__ == '__main__':
    # Sample test with stride=2
    x = torch.randn(1, 3, 32, 32).cuda()
    model = ECB(3, 8, 2, stride=1, act_type='linear', with_idt=True).cuda()
    y_train = model(x)  # training forward
    print("Training output shape:", y_train.shape)

    # Compare with re-param forward
    model.eval()
    with torch.no_grad():
        y_infer = model(x)
    print("Inference output shape:", y_infer.shape)
    print("Difference (training - inference):", (y_train - y_infer).abs().mean())
