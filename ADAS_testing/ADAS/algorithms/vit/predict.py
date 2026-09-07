"""
predict.py - stand-alone ViT inference (image / video / webcam).

Usage:
    python algorithms/vit/predict.py --image path/to/img.jpg
    python algorithms/vit/predict.py --image path/to/img.jpg --draw  # save annotated image
    python algorithms/vit/predict.py --source 0                     # webcam
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing.config import WEATHER_CLASSES
from . import config
from .model import ViTPredictor


def annotate_frame(bgr, result, font_scale=0.7, thickness=2):
    out = bgr.copy()
    h, w = out.shape[:2]
    pred = result.get("prediction", "?")
    conf = result.get("confidence", 0.0)
    lab = f"{pred}  {conf*100:.1f}%"
    cv2.rectangle(out, (8, 8), (260, 48), (0, 0, 0), -1)
    cv2.putText(out, lab, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 0), thickness, cv2.LINE_AA)
    return out


def run_image(image_path, pred, out_dir):
    result = pred.predict(image_path)
    print(json.dumps(result, indent=2))
    bgr = cv2.imread(str(image_path))
    annotated = annotate_frame(bgr, result)
    out = Path(out_dir) / (Path(image_path).stem + "_vit.png")
    cv2.imwrite(str(out), annotated)
    print(f"saved annotated image -> {out}")


def run_stream(src, pred, out_dir, is_webcam=False):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {src}")
    writer = None
    frame_count, total_ms = 0, 0.0
    print(f"processing {src} ... press ESC to stop (webcam).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = pred.predict(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        total_ms += dt
        frame_count += 1
        annotated = annotate_frame(frame, result)
        if writer is None:
            h, w = annotated.shape[:2]
            stem = "webcam" if is_webcam else Path(src).stem
            out = Path(out_dir) / f"{stem}_vit.mp4"
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
        writer.write(annotated)
        if is_webcam:
            cv2.imshow("ViT predictor", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    if writer is not None:
        writer.release()
    if is_webcam:
        cv2.destroyAllWindows()
    if frame_count:
        avg = total_ms / frame_count
        print(f"{frame_count} frames, avg {avg:.1f} ms ({1000.0/avg:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Standalone ViT inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--source", type=str)
    parser.add_argument("--weights", type=str, default=str(config.MODEL_OUT / "best.pt"))
    parser.add_argument("--out",    type=str, default=str(config.RESULTS_OUT))
    args = parser.parse_args()

    pred = ViTPredictor(weights_path=args.weights, num_classes=config.NUM_CLASSES,
                        class_names=WEATHER_CLASSES)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        run_image(args.image, pred, out_dir)
    else:
        run_stream(args.source, pred, out_dir, is_webcam=args.source.isdigit())


if __name__ == "__main__":
    main()
