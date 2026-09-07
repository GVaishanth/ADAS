"""
train.py - train TrafficSignClassifier on GTSRB (or torchvision's auto-download).

Usage:
    python algorithms/traffic_sign/train.py --gtsrb-root datasets/gtsrb
    python algorithms/traffic_sign/train.py  # uses torchvision.datasets.GTSRB to auto-download
"""

import argparse, json, sys, time
from pathlib import Path
import torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config
from .dataset import get_loaders
from .model import TrafficSignClassifier


def _train_one(model, loader, optimizer, criterion, device):
    model.train(); t_loss=correct=total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        t_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return t_loss/max(total,1), correct/max(total,1)


@torch.no_grad()
def _eval(model, loader, criterion, device):
    model.eval(); v_loss=correct=total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        v_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return v_loss/max(total,1), correct/max(total,1)


def main():
    p = argparse.ArgumentParser(description="Train Traffic Sign classifier")
    p.add_argument("--gtsrb-root", type=Path, default=config.GTSRB_DIR)
    p.add_argument("--auto-download", action="store_true",
                   help="use torchvision.datasets.GTSRB(download=True) when no folder exists")
    p.add_argument("--epochs",   type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--batch",    type=int, default=config.DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=config.DEFAULT_LR)
    p.add_argument("--pretrained", action="store_true",
                   help="start from ImageNet weights (mixed help - GTSRB is small)")
    args = p.parse_args()

    train_root = args.gtsrb_root
    if not (train_root / "Training").exists():
        if args.auto_download:
            print("GTSRB folder missing - downloading via torchvision (~300 MB).")
            from torchvision.datasets import GTSRB
            train_root.parent.mkdir(parents=True, exist_ok=True)
            # happens inside GTSRB constructor when split='train'
            GTSRB(root=str(train_root.parent), split="train", download=True)
            GTSRB(root=str(train_root.parent), split="test",  download=True)
        else:
            raise SystemExit(f"GTSRB folder not found at {train_root}. "
                              "Download from https://benchmark.ini.rub.de/gtsrb_dataset.html,"
                              " or pass --auto-download.")

    train_loader, val_loader = get_loaders(args.batch, 0, train_root)
    print(f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    device = config.get_device()
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    model = TrafficSignClassifier(num_classes=config.NUM_CLASSES, pretrained=args.pretrained).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(args.epochs):
        t0=time.perf_counter(); tl, ta = _train_one(model, train_loader, opt, crit, device)
        vl, va = _eval(model, val_loader, crit, device)
        print(f"epoch {ep+1}/{args.epochs} train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f}  ({time.perf_counter()-t0:.1f}s)")
        if va > best:
            best = va
            torch.save(model.state_dict(), config.MODEL_OUT / "best.pt")
    json.dump({"args": vars(args), "best_val_acc": best},
              open(config.RESULTS_OUT / "train_log.json", "w"), indent=2)
    print("Best val acc:", best)

if __name__ == "__main__":
    main()
