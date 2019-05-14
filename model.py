import torch.nn as nn
from capsule_layer import CapsuleConv2d, CapsuleLinear


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = CapsuleConv2d(3, 64, 5, 1, 8, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv2 = CapsuleConv2d(64, 64, 5, 8, 8, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv3 = CapsuleConv2d(64, 128, 5, 8, 16, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv4 = CapsuleConv2d(128, 128, 3, 16, 16, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv5 = CapsuleConv2d(128, 256, 3, 16, 32, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv6 = CapsuleConv2d(256, 256, 3, 32, 32, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv7 = CapsuleConv2d(256, 256, 3, 32, 32, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv8 = CapsuleConv2d(256, 512, 3, 32, 64, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv9 = CapsuleConv2d(512, 512, 3, 64, 64, 1, 0, share_weight=False, bias=False, squash=False)
        self.conv10 = CapsuleConv2d(512, 512, 3, 64, 64, 1, 0, share_weight=False, bias=False, squash=False)
        self.classifier = CapsuleLinear(10, 64, 64, in_capsules=288, share_weight=False, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, 64)
        out = self.classifier(out)
        return out.norm(dim=-1)
