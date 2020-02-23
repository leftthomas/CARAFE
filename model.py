import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d


class Model(nn.Module):
    def __init__(self, backbone_type='resnet18', feature_dim=128):
        super(Model, self).__init__()

        # backbone
        backbones = {'resnet18': (resnet18, 1), 'resnet34': (resnet34, 1), 'resnet50': (resnet50, 4),
                     'resnext50': (resnext50_32x4d, 4)}
        backbone, expansion = backbones[backbone_type]

        self.f = []
        for name, module in backbone(pretrained=True).named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512 * expansion, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
