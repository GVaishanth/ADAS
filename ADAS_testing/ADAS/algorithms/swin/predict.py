"""predict.py - Swin inference for image/video/webcam (same UI as ViT)."""

import argparse, json, sys, time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing.config import WEATHER_CLASSES
from . import config
from .model import SwinPredictor

def annotate_frame(bgr, result, frame_id=-1, fps_avg=0.0):
    out = bgr.copy()
    cv2.rectangle(out, (8, 8), (300, 56), (0, 0, 0), -1)
    cv2.putText(out, f"{result['prediction']}  {result['confidence']*100:.1f}%",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Swin | {fps_avg:.1f} FPS",
                (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out

def main():
    p = argparse.ArgumentParser(description="Standalone Swin inference")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str); g.add_argument("--source", type=str)
    p.add_argument("--weights", type=str, default=str(config.MODEL_OUT / "best.pt"))
    p.add_argument("--out",    type=str, default=str(config.RESULTS_OUT))
    args = p.parse_args()
    pred = SwinPredictor(weights_path=args.weights, num_classes=config.NUM_CLASSES,
                         class_names=WEATHER_CLASSES)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        result = pred.predict(args.image)
        print(json.dumps(result, indent=2))
        bgr = cv2.imread(str(args.image))
        cv2.imwrite(str(out_dir / (Path(args.image).stem + "_swin.png")),
                    annotate_frame(bgr, result))
    else:
        is_webcam = args.source.isdigit()
        cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
        if not cap.isOpened(): raise SystemExit(args.source)
        writer = None; frame_count = total_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            t0 = time.perf_counter(); r = pred.predict(frame); dt = (time.perf_counter()-t0)*1000
            total_ms += dt; frame_count += 1
            fps = 1000.0/dt if dt > 0 else 0
            annotated = annotate_frame(frame, r, frame_count,
                                       fps * 0.4 + (1000.0 / (total_ms / frame_count)) * 0.6)
            if writer is None:
                h, w = annotated.shape[:2]
                out = out_dir / (("webcam" if is_webcam else Path(args.source).stem) + "_swin.mp4")
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
            writer.write(annotated)
            if is_webcam:
                cv2.imshow("Swin", annotated)
                if cv2.waitKey(1) & 0xFF == 27: break
        cap.release()
        if writer is not None: writer.release()
        if is_webcam: cv2.destroyAllWindows()
        if frame_count:
            print(f"{frame_count} frames | avg {total_ms/frame_count:.1f} ms | "
                  f"{1000.0*frame_count/total_ms:.1f} FPS")

if __name__ == "__main__":
    main()
