import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class HWConv(nn.Module):
    r"""Applies a factored 2D convolution over an input signal with distinct h and w axes,
    by performing a 1D convolution over the h axis to an intermediate subspace, followed
    by a 1D convolution over the w axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during
              their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(HWConv, self).__init__()

        kernel_size, stride, padding = _pair(kernel_size), _pair(stride), _pair(padding)
        # decomposing the parameters into h and w components by masking out the values with
        # the defaults on the axis that won't be convolved over.
        h_kernel_size, h_stride, h_padding = (kernel_size[0], 1), (stride[0], 1), (padding[0], 0)
        w_kernel_size, w_stride, w_padding = (1, kernel_size[1]), (1, stride[1]), (0, padding[1])

        # compute the number of intermediary channels (M)
        if bias is True:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * in_channels * out_channels) / (
                        1 + kernel_size[0] * in_channels + kernel_size[1] * out_channels)))
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * in_channels * out_channels) / (
                        kernel_size[0] * in_channels + kernel_size[1] * out_channels)))

        self.h_conv = nn.Conv2d(in_channels, intermed_channels, h_kernel_size, stride=h_stride, padding=h_padding,
                                bias=bias)
        self.bn1 = nn.BatchNorm2d(intermed_channels)

        self.w_conv = nn.Conv2d(intermed_channels, out_channels, w_kernel_size, stride=w_stride, padding=w_padding,
                                bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.h_conv(x)))
        x = self.relu(self.bn2(self.w_conv(x)))
        return x


class WHConv(nn.Module):
    r"""Applies a factored 2D convolution over an input signal with distinct w and h axes,
    by performing a 1D convolution over the w axis to an intermediate subspace, followed
    by a 1D convolution over the h axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during
              their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(WHConv, self).__init__()

        kernel_size, stride, padding = _pair(kernel_size), _pair(stride), _pair(padding)
        # decomposing the parameters into w and h components by masking out the values with the
        # defaults on the axis that won't be convolved over.
        w_kernel_size, w_stride, w_padding = (1, kernel_size[1]), (1, stride[1]), (0, padding[1])
        h_kernel_size, h_stride, h_padding = (kernel_size[0], 1), (stride[0], 1), (padding[0], 0)

        # compute the number of intermediary channels (M)
        if bias is True:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * in_channels * out_channels) / (
                        1 + kernel_size[0] * out_channels + kernel_size[1] * in_channels)))
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * in_channels * out_channels) / (
                        kernel_size[0] * out_channels + kernel_size[1] * in_channels)))

        self.w_conv = nn.Conv2d(in_channels, intermed_channels, w_kernel_size, stride=w_stride, padding=w_padding,
                                bias=bias)
        self.bn1 = nn.BatchNorm2d(intermed_channels)

        self.h_conv = nn.Conv2d(intermed_channels, out_channels, h_kernel_size, stride=h_stride, padding=h_padding,
                                bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.w_conv(x)))
        x = self.relu(self.bn2(self.h_conv(x)))
        return x


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=4)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=4)

        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inp):
        x = F.relu(self.bn1(self.conv1(inp)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)
        return x
