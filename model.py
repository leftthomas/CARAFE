
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = ResNet(BasicBlock, [2, 2, 2, 2]), []
        for name, module in basic_model.named_children():
            if isinstance(module, nn.Linear) or isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.fc = nn.Linear(7 * 7 * 512, num_classes)
        # self.fc = CapsuleLinear(out_capsules=16, in_length=64, out_length=32)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
