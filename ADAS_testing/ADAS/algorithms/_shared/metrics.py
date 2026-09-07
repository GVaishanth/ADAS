"""
metrics.py - common classification metrics + confusion-matrix plot.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             precision_score, recall_score, confusion_matrix)


def compute_classification_metrics(y_true, y_pred, class_names):
    """Returns a dict of classification metrics + the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    # use weighted average so minority classes contribute proportionally
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "accuracy": float(acc),
        "precision_macro": float(macro_prec),
        "recall_macro": float(macro_rec),
        "f1_macro": float(macro_f1),
        "precision_per_class": {c: float(p) for c, p in zip(class_names, prec)},
        "recall_per_class":    {c: float(r) for c, r in zip(class_names, rec)},
        "f1_per_class":        {c: float(f) for c, f in zip(class_names, f1)},
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def plot_confusion_matrix(cm, class_names, out_path: Path, normalize=True, title="Confusion matrix"):
    """Save a heatmap of the confusion matrix to out_path."""
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (normalized)" if normalize else ""))
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class AverageMeter:
    """Simple running mean + std helper for timing."""
    def __init__(self):
        self.values = []
    def update(self, x):
        self.values.append(float(x))
    @property
    def avg(self): return float(np.mean(self.values)) if self.values else 0.0
    @property
    def std(self): return float(np.std(self.values))  if self.values else 0.0
    @property
    def count(self): return len(self.values)


def model_size_mb(state_dict_or_path) -> float:
    """Approximate on-disk size of a model state_dict (.pt file)."""
    if isinstance(state_dict_or_path, (str, Path)):
        return Path(state_dict_or_path).stat().st_size / (1024 * 1024)
    total = 0
    for t in state_dict_or_path.values():
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total / (1024 * 1024)
