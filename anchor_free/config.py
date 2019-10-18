from detectron2.config import CfgNode as CN


def add_anchor_free_config(cfg):
    """
    Add config for anchor free heads.
    """
    _C = cfg

    _C.MODEL.ANCHOR_FREE_HEADS = CN()

    _C.MODEL.ANCHOR_FREE_HEADS.NAME = "RepHead"
