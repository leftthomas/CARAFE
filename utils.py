import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


def load_data(data_name, data_type, batch_size, shuffle=True):
    if data_type == 'train':
        transform = transforms.Compose([transforms.Resize(224), transforms.RandomCrop(224), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    data_set = ImageFolder(root='data/{}/{}'.format(data_name, data_type), transform=transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    return data_loader


class ProbAM:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, images):
        image_size = (images.size(-1), images.size(-2))
        features = self.model.features(images)
        out = features.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, self.model.classifier.weight.size(-1))
        out, probs = self.model.classifier(out)
        classes = out.norm(dim=-1)
        prob = (probs * classes.unsqueeze(dim=-1)).sum(dim=1)
        prob = prob.view(prob.size(0), *features.size()[-2:], -1)
        prob = prob.permute(0, 3, 1, 2).sum(dim=1)

        features_heat_maps = []
        for i in range(prob.size(0)):
            img = images[i].detach().cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)
            mask = cv2.resize(prob[i].detach().cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            features_heat_maps.append(
                transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        features_heat_maps = torch.stack(features_heat_maps)
        return features_heat_maps
