"""
train_classifier.py
===================
One shared trainer for ViT / Swin / ConvNeXt / EfficientFormer so the four-way
comparison is apples-to-apples AND uses the small-data recipe (pretrained +
MixUp/CutMix + label smoothing + cosine warmup + early stopping).

Run from ADAS/ (cwd = ADAS/):
    python -m algorithms._shared.train_classifier --backbone vit \
        --epochs 30 --pretrained --augment

The individual `algorithms/<name>/train.py` scripts now just call this so
there is a single source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from algorithms.preprocessing.config import WEATHER_CLASSES
from algorithms.preprocessing.preprocessing import BDDWeatherDataset
from algorithms._shared.training import (get_loaders, set_seed,
                                         train_classifier_loop)
from algorithms._shared.augmentation import (build_augment_pipeline,
                                             build_basic_pipeline,
                                             LabelSmoothCELoss)

BACKBONES = {  # module name -> (classifier factory names)
    "vit": "algorithms.vit",
    "swin": "algorithms.swin",
    "convnext": "algorithms.convnext",
    "efficientformer": "algorithms.efficientformer",
}


# Class names use the upstream (non-capitalized) casing: ViT / Swin /
# ConvNeXt / EfficientFormer. `backbone.capitalize()` would give the wrong
# casing (Vit / Convnext / Efficientformer), so map explicitly.
CLASSIFIER_CLASS = {
    "vit": "ViTClassifier",
    "swin": "SwinClassifier",
    "convnext": "ConvNeXtClassifier",
    "efficientformer": "EfficientFormerClassifier",
}


def get_backbone(backbone: str, num_classes: int, pretrained: bool):
    mod_name = BACKBONES[backbone]
    mod = __import__(f"{mod_name}.model", fromlist=["Classifier"])
    classifier = getattr(mod, CLASSIFIER_CLASS[backbone])
    return classifier(num_classes=num_classes, pretrained=pretrained)


def main():
    ap = argparse.ArgumentParser(description="Train a backbone with the small-data recipe")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="vit")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pretrained", action="store_true",
                    help="load ImageNet weights (recommended for small data — DO use this)")
    ap.add_argument("--augment", action="store_true",
                    help="MixUp/CutMix + color jitter + rotation (recommended)")
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="override checkpoint path")
    args = ap.parse_args()

    mod_name = BACKBONES[args.backbone]
    cfg = __import__(f"{mod_name}.config", fromlist=["config"])
    set_seed(cfg.SEED)

    # ---- datasets: with or without the strong augmentation pipeline ----
    train_tf = build_augment_pipeline() if args.augment else None
    val_tf = build_basic_pipeline()
    train_ds = BDDWeatherDataset(split="train", transform=train_tf)
    val_ds = BDDWeatherDataset(split="val", transform=val_tf)
    print(f"backend={args.backbone} pretrained={args.pretrained} augment={args.augment} "
          f"train={len(train_ds)} val={len(val_ds)}")
    train_loader, val_loader = get_loaders(train_ds, val_ds,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)

    device = cfg.get_device()
    model = get_backbone(args.backbone, cfg.NUM_CLASSES, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=cfg.WEIGHT_DECAY)
    criterion = LabelSmoothCELoss(cfg.NUM_CLASSES, smoothing=args.smoothing)
    # note: when MixUp/CutMix is enabled the loop switches to soft-target CE
    # internally, so the hard-label criterion is only used for val/loss display.

    out = Path(args.out) if args.out else cfg.MODEL_OUT / "best.pt"
    result = train_classifier_loop(
        model, train_loader, val_loader, optimizer, criterion, device,
        WEATHER_CLASSES, out,
        epochs=args.epochs, warmup=args.warmup,
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
        patience=args.patience, seed=cfg.SEED, prefix=args.backbone)

    cfg.RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    with open(cfg.RESULTS_OUT / "train_log.json", "w") as f:
        json.dump({"args": vars(args), **result}, f, indent=2)
    print(f"\n{args.backbone}: best_val_acc={result['best_val_acc']:.4f} "
          f"@ epoch {result['best_epoch']+1}; checkpoint={out}")


if __name__ == "__main__":
    main()
