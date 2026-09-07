"""
train.py - optional YOLOv8 fine-tune on the BDD100K detection subset.

The BDD subset must first be built with bdd_to_yolo.py which produces
a yolo-format dataset directory + a bdd100k.yaml card.

Usage:
    python algorithms/yolo/train.py --data datasets/bdd100k_yolo/bdd100k.yaml
    python algorithms/yolo/train.py --data datasets/bdd100k_yolo/bdd100k.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config


def main():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on BDD100K detection")
    p.add_argument("--data",    type=str, required=True,
                   help="path to dataset yaml built by bdd_to_yolo.py")
    p.add_argument("--weights", type=str, default=config.DEFAULT_WEIGHTS,
                   help="starting weights (yolov8n.pt / s/l/x)")
    p.add_argument("--epochs",  type=int, default=30)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--project", type=str, default=str(config.RESULTS_OUT / "train"))
    p.add_argument("--name",    type=str, default="yolov8_bdd")
    args = p.parse_args()

    from ultralytics import YOLO
    config.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.name)
    print("Training done. Best weights saved under the project/name dir.")

if __name__ == "__main__":
    main()
