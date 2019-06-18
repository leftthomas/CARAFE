import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
        image_size = (images.size(-2), images.size(-1))
        features = self.model.features(images)
        out = features.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, self.model.classifier.weight.size(-1))
        out, probs = self.model.classifier(out)
        classes = out.norm(dim=-1)

        probs = (probs * classes.unsqueeze(dim=-1)).sum(dim=1)
        probs = probs.view(probs.size(0), *features.size()[-2:], -1)
        probs = probs.permute(0, 3, 1, 2).sum(dim=1, keepdim=True)
        masks = F.interpolate(probs, image_size, mode='bicubic', align_corners=True)
        masks = masks.view(masks.size(0), 1, -1) - masks.view(masks.size(0), 1, -1).min(dim=-1, keepdim=True)[0]
        masks = (masks / masks.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)).view(masks.size(0), 1, *image_size)

        heat_maps = []
        for img, mask in zip(images, masks):
            # change image to BGR
            img = np.ascontiguousarray(img.detach().cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])
            mask = np.ascontiguousarray(mask.detach().cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32(np.uint8(img * 255))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            # revert image to RGB
            heat_maps.append(torch.from_numpy(np.ascontiguousarray(cam[:, :, ::-1].transpose((2, 0, 1)))))
        heat_maps = torch.stack(heat_maps)
        return heat_maps
