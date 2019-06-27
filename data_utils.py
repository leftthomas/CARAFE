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
            a list containing lists of bounding boxes  [bbox coords, class name]
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
            return res
