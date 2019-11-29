from detectron2.config import CfgNode as CN


def add_anchor_free_config(cfg):
    """
    Add config for anchor free heads.
    """
    _C = cfg

    _C.MODEL.ANCHOR_FREE_HEADS = CN()

    _C.MODEL.ANCHOR_FREE_HEADS.NAME = "RepHead"
    # Number of foreground classes
    _C.MODEL.ANCHOR_FREE_HEADS.NUM_CLASSES = 80
    # Names of the input feature maps to be used by anchor free heads
    # Currently all heads (box, mask, ...) use the same input feature map list
    # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
    _C.MODEL.ANCHOR_FREE_HEADS.IN_FEATURES = ["res4"]
