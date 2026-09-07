"""Algorithm 9 - Semantic segmentation (SegFormer, Cityscapes-pretrained)."""

from .model  import SegmentationPredictor
from .config import CITYSCAPES_CLASSES, CITYSCAPES_COLORMAP

__all__ = ["SegmentationPredictor", "CITYSCAPES_CLASSES", "CITYSCAPES_COLORMAP"]
