"""
dataset.py - GTSRB image-folder loader.

Expected folder layout (as unzipped from the official archive):
    datasets/gtsrb/
        Training/<class_id>/<file>.ppm|.png
        Testing/<class_id>/<file>.ppm|.png

We do NOT auto-download. Use either:
    - Full GTSRB:  https://benchmark.ini.rub.de/gtsrb_dataset.html
    - torchvision.datasets.GTSRB(root=..., download=True)   (auto downloads ~300 MB)

Provides: train_loader / val_loader factories for train.py.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import config


def get_transforms(img_size=None):
    img_size = img_size or config.IMG_SIZE
    mean = [0.3403, 0.3121, 0.3214]
    std  = [0.2724, 0.2608, 0.2669]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def get_datasets(gtsrb_root: Path = None):
    gtsrb_root = Path(gtsrb_root) if gtsrb_root else config.GTSRB_DIR
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(gtsrb_root / "Training",  transform=train_tf)
    val_ds   = datasets.ImageFolder(gtsrb_root / "Testing",   transform=val_tf)
    return train_ds, val_ds


def get_loaders(batch_size=None, num_workers=0, gtsrb_root: Path = None):
    train, val = get_datasets(gtsrb_root)
    bs = batch_size or config.DEFAULT_BATCH
    return (
        DataLoader(train, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=False),
        DataLoader(val,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=False),
    )
