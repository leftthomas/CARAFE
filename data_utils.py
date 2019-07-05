import os

import numpy as np
import torch
from pycocotools.coco import COCO


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of class names to indexes
    Arguments:
        keep_difficult (bool, optional): keep difficult instances or not (default: False)
    """

    def __init__(self, keep_difficult=False):
        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                       'train', 'tvmonitor']
        self.class_to_ind = dict(zip(class_names, range(len(class_names))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable, will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        res = []
        if isinstance(target['annotation']['object'], dict):
            objects = [target['annotation']['object']]
        else:
            objects = target['annotation']['object']

        height, width = int(target['annotation']['size']['height']), int(target['annotation']['size']['width'])
        for obj in objects:
            difficult = int(obj['difficult']) == 1
            if not self.keep_difficult and difficult:
                continue

            bbox, pts, bndbox = obj['bndbox'], ['xmin', 'ymin', 'xmax', 'ymax'], []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[pt]) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[obj['name']]
            # [xmin, ymin, xmax, ymax, label_ind]
            bndbox.append(label_idx)
            # [[xmin, ymin, xmax, ymax, label_ind], ... ]
            res += [bndbox]
        res = np.array(res)
        return res


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initialized with a dictionary lookup of class names to indexes
    Arguments:
        annFile (string): Path to json annotation file
    """

    def __init__(self, annFile):
        self.class_to_ind = {}
        labels = open(os.path.join('data', 'coco_labels.txt'), 'r')
        for line in labels:
            ids = line.split(',')
            self.class_to_ind[int(ids[0])] = int(ids[1])
        self.coco = COCO(annFile)

    def __call__(self, target):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        res = []

        if len(target) == 0:
            return np.array(res)
        image = self.coco.loadImgs(target[0]['image_id'])[0]
        height, width = image['height'], image['width']
        for obj in target:
            bbox = obj['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            bndbox = []
            for i in range(4):
                cur_pt = bbox[i]
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[obj['category_id']] - 1
            # [xmin, ymin, xmax, ymax, label_ind]
            bndbox.append(label_idx)
            # [[xmin, ymin, xmax, ymax, label_ind], ... ]
            res += [bndbox]
        res = np.array(res)
        return res


def collate_fn(batch):
    """ list of tensors to a batch tensors """
    images, boxes, labels = [], [], []
    for sample in batch:
        images.append(sample[0])
        if len(sample[1]) != 0:
            boxes.append(torch.tensor(sample[1][:, :4], dtype=torch.float))
            labels.append(torch.tensor(sample[1][:, 4], dtype=torch.long))
        else:
            boxes.append(torch.tensor([], dtype=torch.float))
            labels.append(torch.tensor([], dtype=torch.long))
    return torch.stack(images, 0), boxes, labels
