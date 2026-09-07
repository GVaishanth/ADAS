"""train.py - fine-tune Swin-Tiny on the BDD weather subset."""

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
from algorithms._shared.training import evaluate_classification, get_loaders, set_seed

from . import config
from .model import SwinClassifier

def _train_one(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y); loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def main():
    p = argparse.ArgumentParser(description="Train Swin on BDD weather subset")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = p.parse_args()

    set_seed(config.SEED)
    device = config.get_device()
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    train_ds = BDDWeatherDataset(split="train", transform=get_train_transforms())
    val_ds   = BDDWeatherDataset(split="val",   transform=get_val_transforms())
    print(f"train: {len(train_ds)} images | val: {len(val_ds)} images")
    train_loader, val_loader = get_loaders(train_ds, val_ds,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers)
    model = SwinClassifier(num_classes=config.NUM_CLASSES, pretrained=args.pretrained)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _train_one(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, _, _, _ = evaluate_classification(model, val_loader, criterion, device)
        wall = time.perf_counter() - t0
        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}  ({wall:.1f}s)")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), config.MODEL_OUT / "best.pt")
    json.dump({"args": vars(args), "best_val_acc": best_acc},
              open(config.RESULTS_OUT / "train_log.json", "w"), indent=2)
    print("Best val acc:", best_acc)
    print("Run evaluate.py next.")

if __name__ == "__main__":
    main()
