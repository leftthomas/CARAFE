import torch.nn as nn
from capsule_layer import CapsuleConv2d, CapsuleLinear

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = CapsuleLinear(10, 128, 64)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out.norm(dim=-1)

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        in_length = 3
        out_length = 8
        for x in cfg:
            if x == 'M':
                layers += [CapsuleConv2d(in_channels, in_channels, 3, out_length, out_length, 2, 1)]
                out_length = out_length * 2
            else:
                layers += [CapsuleConv2d(in_channels, x, 3, in_length, out_length, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                in_length = out_length
                out_length = in_length
        return nn.Sequential(*layers)

