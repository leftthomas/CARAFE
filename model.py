
import torch.nn as nn
from capsule_layer import CapsuleLinear
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
        self.fc = CapsuleLinear(out_capsules=num_classes, in_length=512, out_length=64, in_capsules=49,
                                share_weight=False, num_iterations=3, bias=False, return_prob=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1, 512)
        out, prob = self.fc(x)
        return out
