import os
import os.path as osp
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.utils.data as data
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


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): image set to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2012')
    """

    def __init__(self, root, image_set=[('2007', 'trainval'), ('2012', 'trainval')], transform=None,
                 target_transform=VOCAnnotationTransform(), dataset_name='VOC2012'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.anno_path = osp.join('%s', 'Annotations', '%s.xml')
        self.img_path = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_set:
            root_path = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(root_path, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((root_path, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self.anno_path % img_id).getroot()
        img = cv2.imread(self.img_path % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self.img_path % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self.anno_path % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        return torch.tensor(self.pull_image(index)).unsqueeze_(0)


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
