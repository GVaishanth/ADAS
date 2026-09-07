"""
predict.py - run YOLODetector on image / video / webcam.

Usage:
    python algorithms/yolo/predict.py --image path/to/img.jpg
    python algorithms/yolo/predict.py --source 0   # webcam
    python algorithms/yolo/predict.py --weights models/yolo/.../best.pt
"""

import argparse, json, sys, time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config
from .model import YOLODetector


def main():
    p = argparse.ArgumentParser(description="YOLOv8 standalone inference")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",  type=str); g.add_argument("--source", type=str)
    p.add_argument("--weights", type=str, default=config.DEFAULT_WEIGHTS)
    p.add_argument("--conf",   type=float, default=config.CONF_THRESHOLD)
    p.add_argument("--all-classes", action="store_true",
                   help="show every COCO class (default keeps only road-relevant)")
    p.add_argument("--out",    type=str, default=str(config.RESULTS_OUT))
    args = p.parse_args()

    detector = YOLODetector(weights=args.weights, conf=args.conf, road_only=not args.all_classes)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        out = detector.predict(args.image)
        print(json.dumps(out, indent=2))
        if out["results"].get("annotated_path") is None:
            # save annotated image now
            save_path = out_dir / (Path(args.image).stem + "_yolo.png")
            detector.predict(args.image, save_annotated_path=save_path)
            print(f"annotated saved -> {save_path}")
    else:
        is_webcam = args.source.isdigit()
        cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
        if not cap.isOpened(): raise SystemExit(args.source)
        writer = None; frame_count = total_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            t0 = time.perf_counter()
            r = detector.predict(frame)
            dt = (time.perf_counter()-t0)*1000
            total_ms += dt; frame_count += 1
            # draw using ultralytics plot util
            results = detector.model.predict(source=frame, conf=detector.conf,
                                              iou=detector.iou, verbose=False)
            annotated = results[0].plot() if results else frame
            cv2.putText(annotated,
                        f"YOLO | {len(r['results']['detections'])} objs | "
                        f"{r['inference_time_ms']:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if writer is None:
                h, w = annotated.shape[:2]
                out = out_dir / (("webcam" if is_webcam else Path(args.source).stem) + "_yolo.mp4")
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
            writer.write(annotated)
            if is_webcam:
                cv2.imshow("YOLO", annotated)
                if cv2.waitKey(1) & 0xFF == 27: break
        cap.release()
        if writer: writer.release()
        if is_webcam: cv2.destroyAllWindows()
        if frame_count:
            print(f"{frame_count} frames | avg {total_ms/frame_count:.1f}ms | "
                  f"{1000.0*frame_count/total_ms:.1f} FPS")


if __name__ == "__main__":
    main()
