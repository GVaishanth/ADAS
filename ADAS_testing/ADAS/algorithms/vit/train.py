"""
train.py - fine-tune ViT on the BDD100K weather subset.

Usage:
    python algorithms/vit/train.py --epochs 5 --batch-size 32
    python algorithms/vit/train.py --epochs 5 --pretrained          # download ImageNet weights first

Saves:  models/vit/best.pt
Appends results to results/comparison.json (a TBD row until you actually train).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing.config import WEATHER_CLASSES
from algorithms.preprocessing.preprocessing import BDDWeatherDataset
from algorithms.preprocessing.preprocessing import get_train_transforms, get_val_transforms
from algorithms._shared.training import (evaluate_classification, get_loaders, set_seed)

from . import config
from .model import ViTClassifier


def main():
    parser = argparse.ArgumentParser(description="Train ViT on BDD weather subset")
    parser.add_argument("--epochs",           type=int,   default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",       type=int,   default=config.DEFAULT_BATCH)
    parser.add_argument("--lr",               type=float, default=config.DEFAULT_LR)
    parser.add_argument("--pretrained",       action="store_true",
                        help="load pretrained ImageNet weights (downloads ~330 MB the first time)")
    parser.add_argument("--num-workers",      type=int,   default=config.NUM_WORKERS)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = config.get_device()
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    # Build datasets using Algorithm 1's manifests + transforms
    train_ds = BDDWeatherDataset(split="train", transform=get_train_transforms())
    val_ds   = BDDWeatherDataset(split="val",   transform=get_val_transforms())
    print(f"train: {len(train_ds)} images  |  val: {len(val_ds)} images")

    train_loader, val_loader = get_loaders(train_ds, val_ds,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)

    model = ViTClassifier(num_classes=config.NUM_CLASSES, pretrained=args.pretrained)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    log_rows = []
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _train_one(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, lat = evaluate_classification(
            model, val_loader, criterion, device)
        wall = time.perf_counter() - t0
        log_rows.append({"epoch": epoch,
                          "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_loss": val_loss,   "val_acc": val_acc,
                          "wall_seconds": round(wall, 2)})
        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"({wall:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_OUT / "best.pt")
            print(f"  -> saved best checkpoint (val_acc={val_acc:.4f})")

    # persist training history
    with open(config.RESULTS_OUT / "train_log.json", "w") as f:
        json.dump({"args": vars(args), "log": log_rows,
                   "best_val_acc": best_acc}, f, indent=2)

    # current run summary - DO NOT fabricate numbers; mark TBD until evaluated
    summary = {
        "model": "vit",
        "timm_model_name": config.TIMM_MODEL,
        "epochs_run": args.epochs,
        "best_val_acc": round(best_acc, 4),
        "checkpoint":   str(config.MODEL_OUT / "best.pt"),
    }
    print("Training finished. Best val accuracy:", best_acc)
    print("Run algorithms/vit/evaluate.py to populate the comparison table.")
    print("Summary:", json.dumps(summary, indent=2))


def _train_one(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


if __name__ == "__main__":
    main()
