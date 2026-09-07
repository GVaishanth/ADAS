"""evaluate.py - run Swin on test split + append its row to comparison table."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing.config import WEATHER_CLASSES
from algorithms.preprocessing.preprocessing import (BDDWeatherDataset,
                                                   get_val_transforms)
from algorithms._shared.metrics import (compute_classification_metrics,
                                       model_size_mb, plot_confusion_matrix)
from algorithms._shared.training import evaluate_classification, get_loaders, set_seed

from . import config
from .model import SwinClassifier

def main():
    p = argparse.ArgumentParser(description="Evaluate Swin on BDD test split")
    p.add_argument("--weights", type=str, default=str(config.MODEL_OUT / "best.pt"))
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH)
    args = p.parse_args()

    set_seed(config.SEED)
    device = config.get_device()
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    test_ds = BDDWeatherDataset(split="test", transform=get_val_transforms())
    _, test_loader = get_loaders(None, test_ds, batch_size=args.batch_size,
                                  num_workers=config.NUM_WORKERS)
    model = SwinClassifier(num_classes=config.NUM_CLASSES)
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print(f"loaded checkpoint: {args.weights}")
    else:
        print(f"WARNING: no checkpoint found at {args.weights} - using random weights")
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    _tl, acc, yt, yp, lat = evaluate_classification(model, test_loader, criterion,
                                                     device, list(test_ds.label_to_idx))
    metrics = compute_classification_metrics(yt, yp, WEATHER_CLASSES)
    metrics["latency_ms_per_image"] = round(float(lat), 3)
    metrics["fps_estimated"] = round(1000.0 / float(lat), 2) if lat > 0 else 0.0
    metrics["model_size_mb"] = round(
        model_size_mb(Path(args.weights) if Path(args.weights).exists() else model.state_dict()), 2)
    from algorithms._shared.paths import portable as _portable
    metrics["checkpoint"] = _portable(Path(args.weights) if Path(args.weights).exists()
                                      else args.weights)
    metrics["model"]      = "swin"
    metrics["accuracy"]   = acc
    metrics["note"] = ("fine-tuned on current {Day,Night,Fog,Rain} dataset"
                       if Path(args.weights).exists()
                       else "PIPELINE VALIDATION ONLY - random weights")
    json.dump(metrics, open(config.RESULTS_OUT / "metrics.json", "w"), indent=2)
    plot_confusion_matrix(metrics["confusion_matrix"], WEATHER_CLASSES,
                          config.RESULTS_OUT / "confusion_matrix.png")
    print(f"Swin acc={acc:.4f} f1_macro={metrics['f1_macro']:.4f} "
          f"latency={metrics['latency_ms_per_image']} ms fps={metrics['fps_estimated']}")
    # append to comparison table
    row = {
        "model":             "swin",
        "accuracy":          round(acc, 4),
        "precision":         round(metrics["precision_macro"], 4),
        "recall":            round(metrics["recall_macro"], 4),
        "f1":                round(metrics["f1_macro"], 4),
        "inference_time_ms": round(metrics["latency_ms_per_image"], 3),
        "fps":               round(metrics["fps_estimated"], 2),
        "model_size_mb":     round(metrics["model_size_mb"], 2),
        "checkpoint":        metrics["checkpoint"],
        "note":              metrics["note"],
    }
    if Path(args.weights).exists():
        rows = []
        if config.COMPARISON_FILE.exists():
            try:
                rows = [r for r in json.loads(config.COMPARISON_FILE.read_text())
                        if r.get("model") != "swin"]
            except Exception:
                pass
        rows.append(row)
        config.COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.COMPARISON_FILE.write_text(json.dumps(rows, indent=2))
    else:
        print("skipping comparison table: no trained checkpoint for swin "
              "(random-weights metrics are pipeline-validation only).")

if __name__ == "__main__":
    main()
