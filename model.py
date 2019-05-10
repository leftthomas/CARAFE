import math

import torch.nn as nn
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


class ResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses HWConv or WHConv in the standard ResNet
    block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        conv_type (Module, optional): Type of conv that is to be used to form the block. Default: HWConv
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, conv_type=HWConv, downsample=False):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        kernel_size = _pair(kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        if self.downsample:
            # downsample with stride=2
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, stride=2, bias=False)
            self.downsampleconv = conv_type(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
            self.downsamplebn = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = conv_type(in_channels, out_channels, kernel_size, padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv_type(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class ResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the layer.
        kernel_size (int or tuple): Size of the convolving kernels.
        layer_size (int): Number of blocks to be stacked to form the layer
        block_type (Module, optional): Type of block that is to be used to form the block. Default: HWConv
        downsample (bool, optional): If ``True``, the first block in the layer will implement downsampling. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=HWConv, downsample=False):

        super(ResLayer, self).__init__()

        # implement the first block
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, block_type, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical
            self.blocks += [ResBlock(out_channels, out_channels, kernel_size, block_type)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class FeatureLayer(nn.Module):
    r"""Forms a feature layer by initializing 5 layers, with the number of blocks in each layer set by layer_sizes,
    and by performing a global average pool at the end producing a 512-dimensional vector for each element in the batch.
    Args:
        layer_sizes (tuple): An iterable containing the number of blocks in each layer
        block_type (Module, optional): Type of block that is to be used to form the block. Default: HWConv
    """

    def __init__(self, layer_sizes, block_type=HWConv):
        super(FeatureLayer, self).__init__()

        self.conv1 = block_type(1, 64, (3, 7), stride=(1, 2), padding=(1, 3), bias=False)
        self.conv2 = ResLayer(64, 64, (3, 7), layer_sizes[0], block_type=block_type)
        self.conv3 = ResLayer(64, 128, (3, 7), layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = ResLayer(128, 256, (3, 7), layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = ResLayer(256, 512, (3, 7), layer_sizes[3], block_type=block_type, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    r"""Forms a complete two-stream ResNet classifier producing vectors of size num_classes, by initializing a feature
    layers, and passing them through a Linear layer.
    Args:
        num_classes(int): Number of classes in the data
        layer_sizes (tuple): An iterable containing the number of blocks in each layer
    """

    def __init__(self, num_classes, layer_sizes):
        super(Model, self).__init__()

        # HWConv Stream
        self.feature_hw = FeatureLayer(layer_sizes, block_type=HWConv)
        self.fc_hw = nn.Linear(512, num_classes)
        # WHConv Stream
        self.feature_wh = FeatureLayer(layer_sizes, block_type=WHConv)
        self.fc_wh = nn.Linear(512, num_classes)

        self.__init_weight()

    def forward(self, x):
        # HWConv pipeline
        x_hw = self.feature_hw(x)
        logits_hw = self.fc_hw(x_hw)

        # WHConv pipeline
        x_wh = self.feature_wh(x)
        logits_wh = self.fc_wh(x_wh)

        logits = (logits_hw + logits_wh) / 2
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
