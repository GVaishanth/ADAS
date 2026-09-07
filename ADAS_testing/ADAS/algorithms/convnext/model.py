"""ConvNeXt thin subclass over the shared base."""

from algorithms._shared.transformer_base import TimmClassifier, TimmPredictor
from . import config

class ConvNeXtClassifier(TimmClassifier):
    def __init__(self, num_classes=None, pretrained=False):
        super().__init__(num_classes=num_classes or config.NUM_CLASSES, pretrained=pretrained,
                         model_name=config.TIMM_MODEL)

class ConvNeXtPredictor(TimmPredictor):
    def __init__(self, weights_path=None, num_classes=None, device=None, class_names=None):
        super().__init__(weights_path=weights_path, num_classes=num_classes or config.NUM_CLASSES,
                         model_name=config.TIMM_MODEL,
                         device=device, class_names=class_names)
