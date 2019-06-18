
import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if isinstance(module, nn.Linear) or isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.classifier = CapsuleLinear(out_capsules=num_classes, in_length=64, out_length=32, in_capsules=None,
                                        share_weight=True, num_iterations=3, bias=False, return_prob=True)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, 64)
        out, prob = self.classifier(x)
        out = out.norm(dim=-1)
        return out
