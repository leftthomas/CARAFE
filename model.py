import torch
import torch.nn as nn
from backbone import eca_mobilenet_v2
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, ensemble_size, meta_class_size, feature_dim, share_type='block5'):
        super(Model, self).__init__()

        # configs
        self.ensemble_size = ensemble_size
        module_names = ['conv', 'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'block7', 'last_conv']
        if share_type != 'none':
            common_module_names = module_names[:module_names.index(share_type) + 1]
            individual_module_names = module_names[module_names.index(share_type) + 1:]
        else:
            common_module_names, individual_module_names = [], module_names

        # common features
        self.head = []
        for name, module in eca_mobilenet_v2(pretrained=True, last_channel=feature_dim,
                                             num_classes=meta_class_size).named_modules():
            if name in common_module_names:
                self.head.append(module)
        self.head = nn.Sequential(*self.head)
        print("# trainable common feature parameters:", sum(param.numel() if param.requires_grad else 0 for
                                                            param in self.head.parameters()))

        # individual features
        self.tails = []
        for i in range(ensemble_size):
            tail = []
            for name, module in eca_mobilenet_v2(pretrained=True, last_channel=feature_dim,
                                                 num_classes=meta_class_size).named_modules():
                if name in individual_module_names:
                    tail.append(module)
            tail = nn.Sequential(*tail)
            self.tails.append(tail)
        self.tails = nn.ModuleList(self.tails)
        print("# trainable individual feature parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.tails.parameters()) // ensemble_size)

        self.classifier = nn.ModuleList([nn.Linear(feature_dim, meta_class_size) for _ in range(ensemble_size)])
        print("# trainable individual classifier parameters:",
              sum(param.numel() if param.requires_grad else 0 for param in
                  self.classifier.parameters()) // ensemble_size)

    def forward(self, x):
        shared = self.head(x)
        features, out = [], []
        for i in range(self.ensemble_size):
            feature = self.tails[i](shared)
            feature = torch.flatten(F.adaptive_avg_pool2d(feature, output_size=(1, 1)), start_dim=1)
            features.append(feature)
            classes = self.classifier[i](feature)
            out.append(classes)
        features = F.normalize(torch.stack(features, dim=1), dim=-1)
        out = torch.stack(out, dim=1)
        return features, out
