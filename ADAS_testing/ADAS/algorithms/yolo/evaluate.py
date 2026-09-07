"""
evaluate.py - compute mAP / precision / recall on the BDD-YOLO val split.

Usage:
    python algorithms/yolo/evaluate.py \
            --weights models/yolo/train/yolov8_bdd/weights/best.pt \
            --data    datasets/bdd100k_yolo/bdd100k.yaml
"""

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config


def main():
    p = argparse.ArgumentParser(description="YOLOv8 evaluation (mAP/Precision/Recall)")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data",    type=str, required=True)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  type=str, default="auto")
    p.add_argument("--out",     type=str, default=str(config.RESULTS_OUT))
    args = p.parse_args()

    from ultralytics import YOLO
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    model   = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    # ultralytics 8.x exposes these on metrics.results_dict; results_csv exists too
    summary = {
        "model":         "yolo",
        "weights":       args.weights,
        "mAP50":         float(getattr(metrics.box, "map50",   0.0)),
        "mAP50-95":      float(getattr(metrics.box, "map",      0.0)),
        "precision":     float(getattr(metrics.box, "mp",       0.0)),
        "recall":        float(getattr(metrics.box, "mr",       0.0)),
        "inference_ms":  float(getattr(metrics, "speed", {}).get("inference", 0.0) if hasattr(metrics, "speed") else 0.0),
        "note": "PIPELINE VALIDATION ONLY until you run with real fine-tuned weights",
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
