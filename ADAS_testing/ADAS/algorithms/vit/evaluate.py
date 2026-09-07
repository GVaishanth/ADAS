"""
evaluate.py - run ViTClassifier on the test split and append its row to the
shared results/comparison.json file.

Usage:
    python algorithms/vit/evaluate.py
    python algorithms/vit/evaluate.py --weights models/vit/best.pt
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
from algorithms.preprocessing.preprocessing import (BDDWeatherDataset,
                                                   get_val_transforms)
from algorithms._shared.metrics import model_size_mb
from algorithms._shared.training import evaluate_classification, get_loaders, set_seed
from algorithms._shared.metrics import compute_classification_metrics, plot_confusion_matrix

from . import config
from .model import ViTClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT on BDD test split")
    parser.add_argument("--weights", type=str, default=str(config.MODEL_OUT / "best.pt"))
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = config.get_device()
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    test_ds  = BDDWeatherDataset(split="test", transform=get_val_transforms())
    _, test_loader = get_loaders(None, test_ds, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    model = ViTClassifier(num_classes=config.NUM_CLASSES)
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print(f"loaded checkpoint: {args.weights}")
    else:
        print(f"WARNING: no checkpoint found at {args.weights} - using random weights "
              "(metrics are meaningless, this is pipeline-validation only)")
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    (_t_loss, acc, yt, yp, lat_ms) = evaluate_classification(
        model, test_loader, criterion, device, list(test_ds.label_to_idx))
    metrics = compute_classification_metrics(yt, yp, WEATHER_CLASSES)
    metrics["latency_ms_per_image"] = round(float(lat_ms), 3)
    metrics["fps_estimated"] = round(1000.0 / float(lat_ms), 2) if lat_ms > 0 else 0.0
    metrics["model_size_mb"] = round(
        model_size_mb(Path(args.weights) if Path(args.weights).exists() else model.state_dict()),
        2)
    from algorithms._shared.paths import portable as _portable
    metrics["checkpoint"] = _portable(Path(args.weights) if Path(args.weights).exists()
                                      else args.weights)
    metrics["model"]      = "vit"
    metrics["accuracy"]   = acc
    metrics["val_loss"]   = _t_loss
    metrics["note"] = ("fine-tuned on current {Day,Night,Fog,Rain} dataset"
                       if Path(args.weights).exists()
                       else "PIPELINE VALIDATION ONLY - random weights")

    # write per-model report
    with open(config.RESULTS_OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(metrics["confusion_matrix"], WEATHER_CLASSES,
                          config.RESULTS_OUT / "confusion_matrix.png")
    print(f"ViT  acc={acc:.4f}  f1_macro={metrics['f1_macro']:.4f}  "
          f"latency={metrics['latency_ms_per_image']} ms  "
          f"fps={metrics['fps_estimated']}")
    print(f"saved -> {config.RESULTS_OUT / 'metrics.json'}")
    print(f"saved -> {config.RESULTS_OUT / 'confusion_matrix.png'}")

    # append / replace row in shared comparison table. Only do this for a REAL
    # trained checkpoint; with random weights the numbers are meaningless and
    # would clobber a previously-trained row in the comparison table.
    if Path(args.weights).exists():
        save_comparison_row(metrics)
    else:
        print("skipping comparison table: no trained checkpoint "
              f"(random-weights metrics are pipeline-validation only).")


def save_comparison_row(metrics: dict):
    row = {
        "model":              metrics["model"],
        "accuracy":           round(metrics["accuracy"], 4),
        "precision":          round(metrics["precision_macro"], 4),
        "recall":             round(metrics["recall_macro"], 4),
        "f1":                 round(metrics["f1_macro"], 4),
        "inference_time_ms":  round(metrics["latency_ms_per_image"], 3),
        "fps":                round(metrics["fps_estimated"], 2),
        "model_size_mb":      round(metrics["model_size_mb"], 2),
        "checkpoint":         metrics["checkpoint"],
        "note":               metrics["note"],
    }
    rows = []
    if config.COMPARISON_FILE.exists():
        try:
            existing = json.loads(config.COMPARISON_FILE.read_text())
            rows = [r for r in existing if r.get("model") != row["model"]]
        except Exception:
            rows = []
    rows.append(row)
    config.COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.COMPARISON_FILE.write_text(json.dumps(rows, indent=2))
    print(f"appended to shared comparison -> {config.COMPARISON_FILE}")


if __name__ == "__main__":
    main()
