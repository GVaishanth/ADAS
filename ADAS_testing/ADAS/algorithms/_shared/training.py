"""
training.py - common training / fine-tuning helpers.

Every comparative model (ViT, Swin, ConvNeXt, EfficientFormer, traffic sign,
segmentation) uses the same get_loaders / train_one_epoch / evaluate pipeline
so results are directly comparable.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter, compute_classification_metrics


def set_seed(seed: int = 42):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(train_ds, val_ds, batch_size=32, num_workers=0):
    """Two DataLoaders with sensible defaults for laptop / Colab.

    `train_ds` may be None (e.g. during evaluation, when only the test set is
    loaded); in that case the train loader is returned as None.
    """
    train_loader = None
    if train_ds is not None:
        # Drop the last partial batch only when doing so keeps at least one batch;
        # otherwise reduce the batch size so the loader is never empty.
        tbatch = batch_size
        if len(train_ds) >= batch_size:
            drop = True
        else:
            tbatch = max(1, len(train_ds))
            drop = False
        train_loader = DataLoader(train_ds, batch_size=tbatch, shuffle=True,
                                  num_workers=num_workers, pin_memory=False, drop_last=drop)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    meter = AverageMeter()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        meter.update(loss.item() * x.size(0))
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device, class_names=None):
    """Common classification evaluation; returns (loss, accuracy, y_true, y_pred, latency)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    lat = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        t0 = time.perf_counter()
        out = model(x)
        lat.update((time.perf_counter() - t0) * 1000.0 / x.size(0))
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, y_true, y_pred, lat.avg


# --------------------------------------------------------------------------- #
#  Improved training for the small-data regime.
#  Recipe: ImageNet-pretrained + heavy augmentation (MixUp/CutMix) + label
#  smoothing + cosine-with-warmup + early stopping. See augmentation.py.
# --------------------------------------------------------------------------- #
from .augmentation import LabelSmoothCELoss, mixup, cutmix  # noqa: E402


def train_one_epoch_regularized(model, loader, optimizer, criterion, device,
                                mixup_alpha=0.2, cutmix_alpha=0.2,
                                mixup_p=0.4, cutmix_p=0.4, smoothing=0.1, total=None):
    """One epoch with optional MixUp/CutMix and label smoothing.

    Returns (avg_loss, accuracy). `mixing` produces soft class-probability
    targets, so loss is computed with F.cross_entropy(logits, soft_targets);
    otherwise the provided (hard-label) criterion is used.
    """
    model.train()
    total_loss, correct, total_n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        num_classes = logits.size(1)   # model's real output width (bind one-hot to this)
        if mixup_alpha > 0 or cutmix_alpha > 0:
            xm, ysoft = (cutmix(x, y, alpha=cutmix_alpha, p=cutmix_p, num_classes=num_classes)
                         if random.random() < 0.5
                         else mixup(x, y, alpha=mixup_alpha, p=mixup_p, num_classes=num_classes))
            # recompute logits on mixed input
            logits = model(xm)
            loss = _soft_loss(logits, ysoft, smoothing)
        else:
            loss = criterion(logits, y)
        loss.backward()
        # simple gradient clipping keeps large-transformers stable
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


def _soft_loss(logits, soft_targets, smoothing=0.1):
    """Cross-entropy with float class-probability targets + label smoothing."""
    c = logits.size(1)
    if smoothing > 0:
        soft_targets = (1 - smoothing) * soft_targets + smoothing / c
    logp = torch.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


def build_cosine_warmup(optimizer, total_epochs, warmup_epochs=1, min_lr_ratio=0.0):
    """LambdaLR for linear warmup then cosine decay over the epochs."""
    import torch.optim as optim
    lr0 = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        prog = min(max(prog, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * prog))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_classifier_loop(model, train_loader, val_loader, optimizer, criterion,
                          device, class_names, out_model_path,
                          epochs=25, warmup=1, mixup_alpha=0.2, cutmix_alpha=0.2,
                          patience=5, seed=42, prefix="best"):
    """Train with the small-data recipe, early-stop on val accuracy, save best.

    Returns dict {best_epoch, best_val_acc, history:[...] }.
    """
    set_seed(seed)
    best_acc, best_epoch, bad = 0.0, -1, 0
    history = []
    sched = build_cosine_warmup(optimizer, epochs, warmup_epochs=warmup)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch_regularized(
            model, train_loader, optimizer, criterion, device,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha)
        sched.step()
        val_loss, val_acc, yt, yp, lat = evaluate_classification(
            model, val_loader, criterion, device, class_names)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(val_loss, 4),
                        "val_acc": round(val_acc, 4), "lr": sched.get_last_lr()[0],
                        "latency_ms": round(lat, 2),
                        "wall_seconds": round(time.perf_counter() - t0, 2)})
        print(f"epoch {epoch+1}/{epochs}  train={tr_acc:.4f}  val={val_acc:.4f} "
              f"lat={lat:.1f}ms  ({time.perf_counter()-t0:.1f}s)")
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            torch.save(model.state_dict(), out_model_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop @ epoch {epoch+1} (best val_acc={best_acc:.4f})")
                break
    return {"best_epoch": best_epoch, "best_val_acc": float(best_acc), "history": history}


def classify_with_metadata(model, loader, criterion, device, class_names,
                           save_dir: Path, prefix: str, fps_batch_size: int = 32):
    """Full evaluation: metrics + confusion matrix plot + latency stats.

    Returns a `metrics_dict` so the caller can also append it to the
    shared comparison table.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    avg_loss, acc, yt, yp, lat_ms = evaluate_classification(
        model, loader, criterion, device, class_names)
    metrics = compute_classification_metrics(yt, yp, class_names)
    metrics["val_loss"] = float(avg_loss)
    metrics["latency_ms_per_image"] = float(lat_ms)
    metrics["fps_estimated"] = round(1000.0 / lat_ms, 2) if lat_ms > 0 else 0.0
    metrics["batch_fps_estimated"] = round(1000.0 * fps_batch_size / (lat_ms * fps_batch_size), 2)
    # write report
    with open(save_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    from .metrics import plot_confusion_matrix
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
                          save_dir / f"{prefix}_confusion_matrix.png")
    return metrics
