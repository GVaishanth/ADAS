"""predict.py - segmentation inference for image / video / webcam."""

import argparse, json, sys, time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config
from .model import SegmentationPredictor


def main():
    p = argparse.ArgumentParser(description="Semantic segmentation inference")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str); g.add_argument("--source", type=str)
    p.add_argument("--out", type=str, default=str(config.RESULTS_OUT))
    args = p.parse_args()

    seg = SegmentationPredictor()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        save_overlay = out_dir / (Path(args.image).stem + "_seg.png")
        save_mask   = out_dir / (Path(args.image).stem + "_segmask.png")
        r = seg.predict(args.image, save_overlay_path=save_overlay, save_mask_path=save_mask)
        print(json.dumps(r, indent=2))
    else:
        is_webcam = args.source.isdigit()
        cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
        if not cap.isOpened(): raise SystemExit(args.source)
        writer = None; frame_count = total_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            t0 = time.perf_counter(); r = seg.predict(frame); dt = (time.perf_counter()-t0)*1000
            total_ms += dt; frame_count += 1
            # render overlay for saving
            overlay = frame  # placeholder
            try:
                from .model import SegmentationPredictor as SP
                segp = seg  # already loaded
                overlay = segp._colourise(
                    cv2.imread(r["results"]["mask_path"])[...,0] if r["results"]["mask_path"] and Path(r["results"]["mask_path"]).exists() else (cv2.inRange(frame, 0, 255)*0)[...,0],
                    frame, alpha=0.5)
            except Exception:
                overlay = frame
            cv2.putText(overlay,
                        f"SegFormer | {r['results']['num_classes']} cls | "
                        f"{r['inference_time_ms']:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # build mask save per-frame would be costly; only save overlay
            if writer is None:
                h, w = overlay.shape[:2]
                out = out_dir / (("webcam" if is_webcam else Path(args.source).stem) + "_seg.mp4")
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
            writer.write(overlay)
            if is_webcam:
                cv2.imshow("Segmentation", overlay)
                if cv2.waitKey(1) & 0xFF == 27: break
        cap.release()
        if writer: writer.release()
        if is_webcam: cv2.destroyAllWindows()
        if frame_count:
            print(f"{frame_count} frames | avg {total_ms/frame_count:.1f}ms | "
                  f"{1000.0*frame_count/total_ms:.1f} FPS")

if __name__ == "__main__":
    main()
