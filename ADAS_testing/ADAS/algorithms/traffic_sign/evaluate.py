"""
evaluate.py - run TrafficSignClassifier on the GTSRB test split.

Outputs:  results/traffic_sign/metrics.json
          results/traffic_sign/confusion_matrix.png

DOES NOT fabricate metrics until you run with a trained checkpoint.
"""

import argparse, json, sys
from pathlib import Path
import torch, torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config
from .dataset import get_loaders
from .model import TrafficSignClassifier, GTSRB_CLASS_NAMES
from algorithms._shared.metrics import (compute_classification_metrics,
                                        model_size_mb, plot_confusion_matrix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gtsrb-root", type=Path, default=config.GTSRB_DIR)
    p.add_argument("--weights", type=str, default=str(config.MODEL_OUT / "best.pt"))
    p.add_argument("--batch",   type=int, default=config.DEFAULT_BATCH)
    args = p.parse_args()

    _, val_loader = get_loaders(args.batch, 0, args.gtsrb_root)
    device = config.get_device()
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    model = TrafficSignClassifier(num_classes=config.NUM_CLASSES)
    if Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
        print("loaded", args.weights)
    else:
        print(f"WARNING: no checkpoint {args.weights} - using random weights")
    model.to(device).eval()
    crit = nn.CrossEntropyLoss()
    yt, yp = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += crit(out, y).item() * x.size(0)
            yp.extend(out.argmax(1).cpu().tolist())
            yt.extend(y.cpu().tolist())
            total += x.size(0)
    m = compute_classification_metrics(yt, yp, GTSRB_CLASS_NAMES)
    m["checkpoint"] = str(args.weights); m["model"] = "traffic_sign"
    m["accuracy"]   = sum(int(p == t) for p, t in zip(yp, yt)) / max(total, 1)
    m["val_loss"]   = total_loss / max(total, 1)
    m["model_size_mb"] = round(
        model_size_mb(Path(args.weights) if Path(args.weights).exists() else model.state_dict()), 2)
    m["note"] = ("TBD until run on real test set"
                 if Path(args.weights).exists()
                 else "PIPELINE VALIDATION ONLY - random weights, not training")

    json.dump(m, open(config.RESULTS_OUT / "metrics.json", "w"), indent=2)
    plot_confusion_matrix(m["confusion_matrix"], GTSRB_CLASS_NAMES,
                          config.RESULTS_OUT / "confusion_matrix.png", normalize=True)
    print(f"traffic_sign acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
