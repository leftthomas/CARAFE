import torch.nn as nn
from capsule_layer import CapsuleLinear
from torchvision.models.resnet import resnet34


class Model(nn.Module):

    def __init__(self, num_classes, num_iterations=3):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet34(pretrained=True), []
        for name, module in basic_model.named_children():
            if name == 'layer4' or name == 'avgpool' or name == 'fc':
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.in_length, self.out_length = 256, 32
        self.classifier = CapsuleLinear(out_capsules=num_classes, in_length=self.in_length, out_length=self.out_length,
                                        in_capsules=361, share_weight=False, num_iterations=num_iterations)

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, self.in_length)
        out, prob = self.classifier(x)
        out = out.norm(dim=-1)
        return out, prob
