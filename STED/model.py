import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
# from apcan import APCAB, conv
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from PIL import Image

UP = nn.PixelShuffle(2)

def absfft2d(input, gamma=0.1):
    temp = input.permute(0, 3, 1, 2)
    temp_complex = torch.complex(temp, torch.zeros_like(temp))
    fft_result = torch.fft.fft2(temp_complex)
    absfft = torch.pow(torch.abs(fft_result) + 1e-8, gamma)
    output = absfft.permute(0, 2, 3, 1)
    return output


def fft2d(input):
    fft_out = torch.fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out


def fftshift2d(input):
    b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    output = torchvision.transforms.Resize((128, 128), interpolation=Image.BICUBIC)(output)
    return output


class FCALayer(nn.Module):
    def __init__(self, channel=128, reduction=16, size_psc=128):
        super(FCALayer, self).__init__()
        self.gamma = 0.8
        self.size_psc = size_psc
        self.channel = channel
        self.reduction = reduction
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv3 = nn.Conv2d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        absfft1 = absfft2d(x)
        absfft1 = fftshift2d(absfft1)
        absfft2 = self.conv1(absfft1)

        W = F.adaptive_avg_pool2d(absfft2, 1)
        W = self.conv2(W)
        W = self.conv3(W)
        W = torch.sigmoid(W)

        mul = x * W
        return mul

class SingleConv1(nn.Module):
    def __init__(self, num_layers=100):
        super(SingleConv1, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layers = nn.Sequential(
            *(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) for _ in range(num_layers))
        )
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.layers(out)
        out = UP(out)  # 你需要定义UP函数，不在你提供的代码中
        out = self.tail(out)
        return out


class SingleConv(nn.Module):
    def __init__(self):
        super(SingleConv,self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.singleconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv15 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv16 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv17 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv18 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv19 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.singleconv24 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        out = self.head(x)
        out = self.singleconv1(out)
        out = self.singleconv2(out)
        out = self.singleconv3(out)
        out = self.singleconv4(out)
        out = self.singleconv5(out)
        out = self.singleconv6(out)
        out = self.singleconv7(out)
        out = self.singleconv8(out)
        out = self.singleconv9(out)
        out = self.singleconv10(out)
        out = self.singleconv11(out)
        out = self.singleconv12(out)
        out = self.singleconv13(out)
        out = self.singleconv14(out)
        out = self.singleconv15(out)
        out = self.singleconv16(out)
        out = self.singleconv17(out)
        out = self.singleconv18(out)
        out = self.singleconv19(out)
        out = self.singleconv20(out)
        out = self.singleconv21(out)
        out = self.singleconv22(out)
        out = self.singleconv23(out)
        out = self.singleconv24(out)
        out = UP(out)
        out = self.tail(out)
        return out


class DWConv(nn.Module):
    def __init__(self):
        super(DWConv, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dwconv = DepthWiseConv(n_feat=64)
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.dwconv(out)
        out = UP(out)
        out = self.tail(out)
        return out


class PConv100(nn.Module):
    def __init__(self, num_layers=100):
        super(PConv100, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 定义 Partial_conv3 层的参数字典
        pconv_layer_params = {
            'dim': 64,
            'n_div': 4,
            'forward': 'split_cat'
        }

        # 创建一个包含 num_layers 个 Partial_conv3 层的 nn.Sequential
        self.pconv_layers = nn.Sequential(
            *[Partial_conv3(**pconv_layer_params) for _ in range(num_layers)]
        )

        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.pconv_layers(out)
        out = UP(out)
        out = self.tail(out)
        return out

class PConv(nn.Module):
    def __init__(self):
        super(PConv, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pconv = Partial_conv3(dim=64, n_div= 4, forward='split_cat')
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.pconv(out)
        out = UP(out)
        out = self.tail(out)
        return out


class MultPConv(nn.Module):
    def __init__(self):
        super(MultPConv, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pconv1 = Partial_conv3(dim=64, n_div= 4, forward='split_cat')
        self.pconv2 = Partial_conv3(dim=64, n_div=4, forward='split_cat')
        self.pconv3 = Partial_conv3(dim=64, n_div=4, forward='split_cat')
        self.pconv4 = Partial_conv3(dim=64, n_div=4, forward='split_cat')
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.pconv1(out)
        out = self.pconv2(out)
        out = self.pconv3(out)
        out = self.pconv4(out)
        out = UP(out)
        out = self.tail(out)
        return out

class FZConv(nn.Module):
    def __init__(self):
        super(FZConv, self).__init__()
        self.head = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fz1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,groups=8)
        self.fz2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8)
        self.fz3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8)
        self.fz4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=8)
        self.tail = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.head(x)
        out = self.fz1(out)
        out = self.fz2(out)
        out = self.fz3(out)
        out = self.fz4(out)
        out = UP(out)
        out = self.tail(out)
        return out


class DepthWiseConv(nn.Module):
    def __init__(self, n_feat):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=n_feat,
                                    out_channels=n_feat,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=8)
        self.point_conv = nn.Conv2d(in_channels=n_feat,
                                    out_channels=n_feat,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        # print(out.shape)
        out = self.point_conv(out)
        return out


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


# class DRRN_PConv_AP(nn.Module):
#     def __init__(self):
#         super(DRRN_PConv_AP, self).__init__()
#         self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.pconv1 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.pconv2 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.ca = APCAB(conv=conv)
#         self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.relu = nn.ReLU(inplace=True)
#         self.gelu = nn.GELU()
#
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#
#     def forward(self, x):
#         residual = x  # 1 9 512 512
#         inputs = self.input(self.gelu(x))
#         out = inputs
#         for _ in range(10):
#             out = self.pconv2(self.gelu(self.pconv1(self.gelu(out))))
#             out = torch.add(out, inputs)
#             out = self.ca(out)
#         # out = self.ca(out)
#         # out = self.conv3(self.relu(out))
#         # out = self.ca(out)
#         out = torch.add(out, residual)   # 1 9 1024 1024
#         out = UP(out)
#         out = self.output(self.gelu(out))
#
#         return out

class DRRN_PConv_15(nn.Module):
    def __init__(self):
        super(DRRN_PConv_15, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pconv1 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
        self.pconv2 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
        self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x  # 1 9 512 512
        inputs = self.input(self.gelu(x))
        out = inputs
        for _ in range(15):
            out = self.pconv2(self.gelu(self.pconv1(self.gelu(out))))
            out = torch.add(out, inputs)
            # out = self.ca(out)
        # out = self.conv3(self.relu(out))
        out = torch.add(out, residual)   # 1 9 1024 1024
        out = UP(out)
        out = self.output(self.gelu(out))

        return out

# class DRRN_CA15(nn.Module):
#     def __init__(self):
#         super(DRRN_CA15, self).__init__()
#         self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.ca = CA(channel=128, reduction=16)
#         self.ca = APCAB(conv=conv)
#         # self.conv3 = nn.Conv2d(in_channels=128, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
#         self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#
#     def forward(self, x):
#         residual = x  # 1 9 512 512
#         inputs = self.input(self.relu(x))
#         out = inputs
#         for _ in range(6):
#             out = self.conv2(self.relu(self.conv1(self.relu(out))))
#             out = torch.add(out, inputs)
#             out = self.ca(out)
#         # out = self.conv3(self.relu(out))
#         out = torch.add(out, residual)   # 1 9 1024 1024
#         out = UP(out)
#         out = self.output(self.relu(out))
#
#         return out


# class Double_AP(nn.Module):
#     def __init__(self):
#         super(Double_AP, self).__init__()
#         self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.pconv1 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.pconv2 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.ca1 = APCAB(conv=conv)
#         self.pconv3 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.pconv4 = Partial_conv3(dim=128, n_div= 4, forward='split_cat')
#         self.ca2 = APCAB(conv=conv)
#         self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.relu = nn.ReLU(inplace=True)
#         self.gelu = nn.GELU()
#
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
#
#     def forward(self, x):
#         residual = x  # 1 9 512 512
#         inputs = self.input(self.gelu(x))
#         out = inputs
#         for _ in range(7):
#             out = self.pconv2(self.gelu(self.pconv1(self.gelu(out))))
#             out = torch.add(out, inputs)
#         out = self.ca1(out)
#         for _ in range(7):
#             out = self.pconv4(self.gelu(self.pconv3(self.gelu(out))))
#             out = torch.add(out, inputs)
#         out = self.ca2(out)
#         # out = self.ca(out)
#         # out = self.conv3(self.relu(out))
#         # out = self.ca(out)
#         out = torch.add(out, residual)   # 1 9 1024 1024
#         out = UP(out)
#         out = self.output(self.gelu(out))
#
#         return out


class DRRN_FCA(nn.Module):
    def __init__(self):
        super(DRRN_FCA, self).__init__()
        self.input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.ca = CA(channel=128, reduction=16)
        self.FCA = FCALayer(channel=64)
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        self.tail = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x1, x2):
        Input = torch.cat((x1, x2), dim=1)
        # 两个输入合并为一个张量
        residual = Input  # 1 2 1024 1024
        # print(Input.shape,'-------')
        inputs = self.input(self.relu(Input))
        out = inputs
        for _ in range(10):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
            out = self.FCA(out)
        # out = self.conv3(self.relu(out))
        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        out = self.tail(self.relu(out))


        return out

class DRRN_FCA_Pconv(nn.Module):
    def __init__(self):
        super(DRRN_FCA_Pconv, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pconv1 = Partial_conv3(dim=128, n_div=4, forward='split_cat')
        self.pconv2 = Partial_conv3(dim=128, n_div=4, forward='split_cat')
        # self.ca = CA(channel=128, reduction=16)
        self.FCA = FCALayer(channel=128)
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x  # 1 9 512 512
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(10):
            out = self.pconv2(self.relu(self.pconv1(self.relu(out))))
            out = torch.add(out, inputs)
            out = self.FCA(out)
        # out = self.conv3(self.relu(out))
        out = torch.add(out, residual)   # 1 9 1024 1024
        out = UP(out)
        out = self.output(self.relu(out))

        return out


# from thop import profile
#
# net = DRRN_FCA()  # 定义好的网络模型
# input = torch.randn(1, 1, 1024, 1024)
# input2 = torch.randn(1, 1, 1024, 1024)
# flops, params = profile(net, (input,input2,))
# print('flops: ', flops, 'params: ', params)