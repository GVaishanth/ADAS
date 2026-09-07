"""
predict.py
==========
Standalone inference for Algorithm 1 - test the preprocessing pipeline on a
single image or a video/webcam stream WITHOUT needing any model.

Usage
-----
    python algorithms/preprocessing/predict.py --image path/to/test.jpg
    python algorithms/preprocessing/predict.py --image test.jpg --enhance clahe --label rainy
    python algorithms/preprocessing/predict.py --source test_video.mp4 --enhance denoise
    python algorithms/preprocessing/predict.py --source 0        # webcam index 0

Outputs are written to results/preprocessing/.
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

from algorithms.preprocessing import config
from algorithms.preprocessing.preprocessing import (
    PreprocessingPipeline,
    apply_enhancement,
    tensor_to_image,
)


def save_image_visualization(image_path, label, pipe, out_path):
    """Side-by-side: Original | Enhanced | What-the-model-sees (denormalized)."""
    bgr = cv2.imread(str(image_path))
    enhanced, mode = apply_enhancement(bgr, pipe.enhance)
    tensor = pipe.preprocess_tensor(bgr)
    model_view = tensor_to_image(tensor)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [bgr, enhanced, model_view],
                              ["Original", f"Enhanced ({mode})",
                               "Model input\n(resized + normalized)"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Preprocessing visualization - {Path(image_path).name}  "
                 f"(known label: {label or 'unknown'})", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_on_image(args):
    pipe = PreprocessingPipeline(enhance=args.enhance)
    result = pipe.preprocess(args.image)
    print(json.dumps(result, indent=2))

    config.RESULTS_PREP.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = config.RESULTS_PREP / f"predict_{Path(args.image).stem}_{stamp}.png"
    save_image_visualization(args.image, args.label, pipe, out_path)
    print(f"\nVisualization saved -> {out_path}")


def run_on_stream(args):
    """Video file or webcam: annotate each frame with preprocessing info."""
    pipe = PreprocessingPipeline(enhance=args.enhance)
    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    is_webcam = src.isdigit()
    writer = None
    out_path = None
    frame_count, total_ms = 0, 0.0

    print(f"Processing {src} ... (webcam: press ESC to stop)")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = pipe.preprocess(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        total_ms += dt
        frame_count += 1

        enhanced, mode = apply_enhancement(frame, pipe.enhance)
        if writer is None:
            h, w = enhanced.shape[:2]
            stem = "webcam" if is_webcam else Path(src).stem
            out_path = (config.RESULTS_PREP /
                        f"processed_{stem}_{time.strftime('%H%M%S')}.mp4")
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, (w, h))
        writer.write(enhanced)
        cv2.putText(enhanced,
                    f"prep {dt:.1f} ms | {mode} | {result['results']['tensor_shape']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_webcam:
            cv2.imshow("Preprocessing pipeline", enhanced)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if is_webcam:
        cv2.destroyAllWindows()
    avg_ms = total_ms / max(frame_count, 1)
    print(f"Processed {frame_count} frames, avg {avg_ms:.1f} ms/frame "
          f"({1000.0 / avg_ms:.1f} FPS baseline) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test the preprocessing pipeline standalone")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="path to a single image")
    group.add_argument("--source", type=str, help="video file path or webcam index (0)")
    parser.add_argument("--enhance", choices=["none", "clahe", "denoise"], default="none",
                        help="optional adverse-weather enhancement")
    parser.add_argument("--label", type=str, default=None,
                        help="optional known weather label for the visualization title")
    args = parser.parse_args()

    if args.image:
        run_on_image(args)
    else:
        run_on_stream(args)


if __name__ == "__main__":
    main()
