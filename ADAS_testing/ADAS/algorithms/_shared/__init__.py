"""
_algorithms/_shared package - common helpers reused by ViT, Swin, ConvNeXt,
EfficientFormer, Traffic Sign, Segmentation, etc. Anything imported by more
than one algorithm lives here so the four comparative models stay in sync.
"""

from .transformer_base import TimmClassifier, TimmPredictor
from .metrics import (AverageMeter, compute_classification_metrics,
                      model_size_mb, plot_confusion_matrix)
from .training import (evaluate_classification, get_loaders, set_seed,
                       train_one_epoch)
from .json_contract import make_envelope, timed, error_envelope
from .transform_helper import (BDDWeatherDataset, get_train_transforms,
                              get_val_transforms)

__all__ = [
    "TimmClassifier", "TimmPredictor",
    "AverageMeter", "compute_classification_metrics",
    "model_size_mb", "plot_confusion_matrix",
    "evaluate_classification", "get_loaders", "set_seed", "train_one_epoch",
    "make_envelope", "timed", "error_envelope",
    "BDDWeatherDataset", "get_train_transforms", "get_val_transforms",
]
