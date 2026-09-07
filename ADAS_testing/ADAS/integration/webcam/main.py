"""
main.py - entry point for the webcam pipeline.

Use:
    python integration/webcam/main.py --source 0
    python integration/webcam/main.py --source path/to/video.mp4
    python integration/webcam/main.py --source 0 --no-yolo --no-seg  # lighter pipeline

ALL models are optional - the pipeline loads them ONLY if you pass --model-name flags
or accept the defaults. Models without trained weights will print a warning
but not crash.

All checkpoint paths default to integration/webcam/config.py - single source
of truth. Override any of them on the command line (e.g.
    --weather-weights  models/efficientformer/best.pt
    --yolo-weights    models/yolo/best.pt
    --no-seg          # skip the slow SegFormer step
).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # ADAS/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# All weight paths come from one place: integration/webcam/config.py
from integration.webcam.config import (
    DEFAULT_VIT_WEIGHTS, DEFAULT_SWIN_WEIGHTS, DEFAULT_CONVNEXT_WEIGHTS,
    DEFAULT_EFFFORMER_WEIGHTS, DEFAULT_YOLO_WEIGHTS, DEFAULT_LANE_WEIGHTS,
    DEFAULT_SIGN_WEIGHTS,
)
from integration.webcam.pipeline import WebcamPipeline
from algorithms.vit           import ViTPredictor
from algorithms.swin          import SwinPredictor
from algorithms.convnext      import ConvNeXtPredictor
from algorithms.efficientformer import EfficientFormerPredictor
from algorithms.yolo          import YOLODetector
from algorithms.lane_detection import LaneDetector
from algorithms.traffic_sign  import TrafficSignPredictor
from algorithms.segmentation  import SegmentationPredictor

WEATHER_PREDICTORS = {
    "vit":            ViTPredictor,
    "swin":           SwinPredictor,
    "convnext":       ConvNeXtPredictor,
    "efficientformer": EfficientFormerPredictor,
}
WEATHER_WEIGHT_FLAGS = {
    "vit":            DEFAULT_VIT_WEIGHTS,
    "swin":           DEFAULT_SWIN_WEIGHTS,
    "convnext":       DEFAULT_CONVNEXT_WEIGHTS,
    "efficientformer": DEFAULT_EFFFORMER_WEIGHTS,
}


def main():
    p = argparse.ArgumentParser(description="Real-time webcam ADAS pipeline")
    p.add_argument("--source", type=str, default="0",
                   help="0 = webcam, otherwise path to a video")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-yolo", action="store_true")
    p.add_argument("--no-seg", action="store_true")
    p.add_argument("--no-signs", action="store_true")
    p.add_argument("--no-lanes", action="store_true")
    p.add_argument("--no-weather", action="store_true")
    # All weight paths default to webcam/config.py
    p.add_argument("--weather-module",    type=str, default="vit",
                   choices=list(WEATHER_PREDICTORS.keys()),
                   help="which transformer to use as the weather model (default: vit)")
    p.add_argument("--weather-weights",   type=str, default=None,
                   help=f"override weather-model checkpoint "
                        f"(default picks the weight for --weather-module; here: "
                        f"vit={DEFAULT_VIT_WEIGHTS}, swin={DEFAULT_SWIN_WEIGHTS}, "
                        f"convnext={DEFAULT_CONVNEXT_WEIGHTS}, "
                        f"efficientformer={DEFAULT_EFFFORMER_WEIGHTS})")
    p.add_argument("--yolo-weights",      type=str, default=str(DEFAULT_YOLO_WEIGHTS),
                   help=f"YOLOv8 weights (default: {DEFAULT_YOLO_WEIGHTS})")
    p.add_argument("--lane-weights",      type=str, default=str(DEFAULT_LANE_WEIGHTS),
                   help="Lane-detection weights (classical mode ignores this)")
    p.add_argument("--sign-weights",      type=str, default=str(DEFAULT_SIGN_WEIGHTS),
                   help=f"traffic-sign weights (default: {DEFAULT_SIGN_WEIGHTS})")
    args = p.parse_args()

    # Pick the weather-model weight: explicit --weather-weights wins;
    # otherwise use the default for the chosen --weather-module.
    weather_weights = args.weather_weights or str(WEATHER_WEIGHT_FLAGS[args.weather_module])

    # Build model instances lazily. We do it OPTIONAL so if the weights
    # don't exist yet we just skip that model gracefully.
    weather_pred = None
    if not args.no_weather:
        try:
            Cls = WEATHER_PREDICTORS[args.weather_module]
            weather_pred = Cls(weights_path=weather_weights, num_classes=6)
        except Exception as e:
            print(f"weather module disabled: {e}")

    obj_det = None
    if not args.no_yolo:
        try:
            obj_det = YOLODetector(weights=args.yolo_weights)
        except Exception as e:
            print(f"yolo disabled: {e}")

    lane_det = None
    if not args.no_lanes:
        try:
            lane_det = LaneDetector(mode="classical", weights=args.lane_weights)
        except Exception as e:
            print(f"lane detector disabled: {e}")

    sign_pred = None
    if not args.no_signs:
        try:
            sign_pred = TrafficSignPredictor(weights_path=args.sign_weights, num_classes=43)
        except Exception as e:
            print(f"sign predictor disabled: {e}")

    seg_pred = None
    if not args.no_seg:
        try:
            seg_pred = SegmentationPredictor()
        except Exception as e:
            print(f"segmentation disabled: {e}")

    pipe = WebcamPipeline(
        weather_pred=weather_pred, object_det=obj_det, lane_det=lane_det,
        sign_pred=sign_pred,        seg_pred=seg_pred,
        enable_weather=not args.no_weather,
        enable_object_detection=not args.no_yolo,
        enable_lanes=not args.no_lanes,
        enable_signs=not args.no_signs,
        enable_segmentation=not args.no_seg,
    )
    pipe.run(source=args.source, max_frames=args.max_frames,
             display=not args.no_display, save_video_path=args.out)


if __name__ == "__main__":
    main()
