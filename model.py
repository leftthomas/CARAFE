import torch.nn as nn
from capsule_layer import CapsuleConv2d


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = CapsuleConv2d(3, 64, 5, 1, 8, 1, 0, share_weight=False, squash=False, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, 5, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 0, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 0, bias=False)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 0, bias=False)
        self.bn10 = nn.BatchNorm2d(512)
        # self.classifier = CapsuleLinear(10, 64, 64, in_capsules=288, share_weight=False, bias=False)
        self.classifier = nn.Linear(18432, 10, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = self.relu(self.bn7(self.conv7(out)))
        out = self.relu(self.bn8(self.conv8(out)))
        out = self.relu(self.bn9(self.conv9(out)))
        out = self.relu(self.bn10(self.conv10(out)))
        out = out.view(out.size(0), -1)
        # out = out.permute(0, 2, 3, 1).contiguous()
        # out = out.view(out.size(0), -1, 64)
        out = self.classifier(out)
        # out = out.norm(dim=-1)
        return out
