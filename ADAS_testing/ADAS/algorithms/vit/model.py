"""
model.py - ViT classifier + predictor (thin subclasses over _shared base).
"""

from algorithms._shared.transformer_base import TimmClassifier, TimmPredictor
from . import config

# Re-export under ViT-specific names so the rest of the module
# (and FastAPI code) can do `from algorithms.vit import ViTClassifier`.
class ViTClassifier(TimmClassifier):
    """ViT-Base-224 fine-tuned for 6-class weather classification."""
    def __init__(self, num_classes=None, pretrained=False):
        super().__init__(num_classes=num_classes or config.NUM_CLASSES, pretrained=pretrained,
                         model_name=config.TIMM_MODEL)


class ViTPredictor(TimmPredictor):
    """ViT predictor with the FastAPI-friendly .predict() JSON contract."""
    def __init__(self, weights_path=None, num_classes=None, device=None, class_names=None):
        super().__init__(weights_path=weights_path, num_classes=num_classes or config.NUM_CLASSES,
                         model_name=config.TIMM_MODEL,
                         device=device, class_names=class_names)
