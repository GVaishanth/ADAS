"""
augmentation.py
===============
Strong regularisation for the four-backbone weather classifiers.

The bundled dataset is small (20 imgs / class, 4 classes). On a tiny dataset the
going-prescription is: ImageNet-pretrained weights + heavy augmentation + label
smoothing + early stopping. This module supplies the augmentation half:

  * build_weather_transforms(phase, size)  -> standard train/val transforms
  * MixUp / CutMix (collate-level, applied to a batch tensor)
  * label smoothing cross entropy

Everything is deliberately kept in numpy/torch so it drops into the existing
DataLoader pipeline (see training.py / preprocessing.py) without touching the
preprocessing contract.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
#  Label smoothing cross-entropy (reduces overconfidence on tiny data)
# --------------------------------------------------------------------------- #
class LabelSmoothCELoss(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al.)."""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        logits = logits.float()
        with torch.no_grad():
            true_dist = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))


# --------------------------------------------------------------------------- #
#  MixUp (mixing inputs and one-hot labels linearly)
# --------------------------------------------------------------------------- #
def mixup(images, targets, alpha=0.2, p=0.5, num_classes=None):
    """Return (mixed_images, soft_targets). Soft targets broadcast to [B, nb_classes]."""
    nb = images.size(0)
    if alpha <= 0 or random.random() > p:
        return images, _onehot(targets, num_classes)

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(nb, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    y_a = _onehot(targets, num_classes)
    y_b = _onehot(targets[idx], num_classes)
    return mixed, lam * y_a + (1 - lam) * y_b


# --------------------------------------------------------------------------- #
#  CutMix (paste a patch from another sample)
# --------------------------------------------------------------------------- #
def cutmix(images, targets, alpha=1.0, p=0.5, num_classes=None):
    """Return (mixed_images, soft_targets) from the patch-based CutMix scheme."""
    nb, c, h, w = images.size()
    if alpha <= 0 or random.random() > p:
        return images, _onehot(targets, num_classes)

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(nb, device=images.device)
    # sample a rectangular box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = random.randint(0, w)
    cy = random.randint(0, h)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx][:, :, y1:y2, x1:x2]

    # lam is recomputed to match the actual patch area
    lam = 1.0 - ((x2 - x1) * (y2 - y1)) / (w * h)
    return mixed, lam * _onehot(targets, num_classes) + (1 - lam) * _onehot(targets[idx], num_classes)


def _onehot(targets: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    """One-hot encode labels to exactly `num_classes` columns.

    IMPORTANT: width must be the model's real number of classes. Deriving it from
    ``targets.max()+1`` is wrong for small batches that don't happen to contain the
    highest-index class (with 4 classes, a batch of labels {0,1,2} gives a 3-wide
    one-hot -> shape mismatch against the model's 4-logit output). Callers must pass
    ``num_classes`` (e.g. logits.size(1)) so every batch is padded consistently.
    """
    if targets.dim() > 1:
        return targets.float()
    if num_classes is None:
        num_classes = int(targets.max().item()) + 1
    return F.one_hot(targets, num_classes=num_classes).float()


# --------------------------------------------------------------------------- #
#  Dataset-level augmentation (applied per RAW PIL image, then tensorised)
# --------------------------------------------------------------------------- #
def strong_color_jitter():
    """Colour jitter for the weather classifiers.

    NOTE: weather is told apart almost entirely by colour and lighting (a clear
    sky is bright + blue, fog is flat/grey, rain is washed/desaturated). Brute-force
    brightness/contrast/saturation jitter at the old 0.35-0.4 destroyed that signal,
    and on the tiny bundled dataset (~14 imgs/class) the model collapsed onto the
    visually-average class ("foggy") for every input. Keep the jitter *mild* so the
    model still sees the lighting cues that define each weather condition while
    retaining a little robustness.
    """
    from torchvision import transforms as T
    return T.ColorJitter(brightness=0.12, contrast=0.10, saturation=0.10, hue=0.02)


def build_augment_pipeline(p: float = 0.8):
    """Compose the stochastic train augmentations on top of the base resize+norm."""
    from torchvision import transforms as T
    from algorithms.preprocessing.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=12, fill=0),
        strong_color_jitter(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_basic_pipeline():
    """Deterministic evaluation (val/test) pipeline — MUST match get_val_transforms().

    Used as the validation transform during training. Keeping it byte-for-byte
    identical to the eval/inference transform (get_val_transforms) is important:
    if train-time validation resized to INPUT_SIZE but inference resized to
    INPUT_SIZE*1.14 then center-cropped, the model sees a different scale at
    inference than it was validated on (inflated train/val vs test mismatch).
    """
    from torchvision import transforms as T
    from algorithms.preprocessing.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Resize(int(INPUT_SIZE * 1.14)),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
