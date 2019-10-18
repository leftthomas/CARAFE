from typing import Dict

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from torch import nn

ANCHOR_FREE_HEADS_REGISTRY = Registry("ANCHOR_FREE_HEADS")


def build_anchor_free_head(cfg, input_shape):
    """
    Build an anchor free head from `cfg.MODEL.ANCHOR_FREE_HEADS.NAME`.
    """
    name = cfg.MODEL.ANCHOR_FREE_HEADS.NAME
    return ANCHOR_FREE_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ANCHOR_FREE_HEADS_REGISTRY.register()
class RepHead(nn.Module):
    """
    A Rep-points head described in detail in the RepPoints: Point Set Representation for Object Detection.
    It takes FPN features as input and merges information from all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        pass

    def forward(self, features, targets=None):
        pass
