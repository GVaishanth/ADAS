"""
ml_service.py
=============
A single lazy-registry that owns every ADAS model and offers high-level
functions the FastAPI endpoints call. Keeping all model construction in one
place (instead of inside each request handler) lets us:

  * load models once and cache them across requests,
  * degrade gracefully when a model can't be built (metadata/weights missing),
  * run one "analyze_media()" that fans a frame out to every enabled task and
    returns a single annotated image + a JSON report.

Task keys
---------
    weather            -> best weather classifier (models/<m>/best.pt if present, else ViT random)
    weather_models     -> {vit,swin,convnext,efficientformer} predictor dict
    objects            -> YOLOv8 detector
    lanes              -> classical OpenCV lane detector
    signs              -> resnet18 traffic-sign classifier
    seg                -> SegFormer semantic segmentation

Everything returns the standard JSON contract
( status / prediction / confidence / inference_time_ms / results ).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2

# Make `algorithms.*` and `integration.*` importable regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]            # ADAS/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path as _P

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Order of the four comparative backbones (for the table + ensemble views).
BACKBONES = ["vit", "swin", "convnext", "efficientformer"]


class _Registry:
    """Builds-and-caches model instances lazily. Thread-safe-enough for a
    single-process uvicorn (the GIL + the fact that build() is idempotent)."""

    def __init__(self):
        self._cache = {}
        self._errors = {}
        self._locked = set()

    # ------------------------------------------------------------------ #
    def build(self, task: str):
        return self._get(task)

    def reset(self, task: str = None):
        if task is None:
            self._cache.clear(); self._errors.clear(); self._locked.clear()
            return
        self._cache.pop(task, None); self._errors.pop(task, None)

    def _get(self, task: str):
        if task in self._cache:
            return self._cache[task]
        if task in self._errors:
            return None
        if task in self._locked:
            # concurrent build of same task in another thread -> wait once
            import time
            n = 0
            while task in self._locked and n < 120:
                time.sleep(0.5); n += 1
            return self._cache.get(task)
        self._locked.add(task)
        try:
            obj = self._build(task)
            if obj is not None:
                self._cache[task] = obj
            else:
                self._errors[task] = "constructor returned None"
        except Exception as e:
            traceback.print_exc()
            self._errors[task] = f"{type(e).__name__}: {e}"
            obj = None
        finally:
            self._locked.discard(task)
        return obj

    # ------------------------------------------------------------------ #
    def _build(self, task: str):
        if task == "weather":
            return self._weather()
        if task == "weather_models":
            return {m: self._weather_backbone(m) for m in BACKBONES}
        if task == "objects":
            return self._objects()
        if task == "lanes":
            return self._lanes()
        if task == "signs":
            return self._signs()
        if task == "seg":
            return self._seg()
        raise ValueError(f"unknown task: {task}")

    # ---- individual builders ------------------------------------------ #
    def _best_ckpt(self, model: str):
        c = MODELS_DIR / model / "best.pt"
        return str(c) if c.exists() else None

    def _weather(self):
        """Best available weather classifier: a trained backbone if present,
        else a small ViT predictor (untrained -> still returns a prediction)."""
        # Prefer a trained checkpoint if any backend trained one.
        for m in BACKBONES:
            ckpt = self._best_ckpt(m)
            if ckpt:
                from algorithms import vit  # generic predictor; pick matching
                mod = _import_mod(m)
                pred = mod.__dict__[{"vit": "ViTPredictor",
                                     "swin": "SwinPredictor",
                                     "convnext": "ConvNeXtPredictor",
                                     "efficientformer": "EfficientFormerPredictor"}[m]]
                return pred(weights_path=ckpt)
        # Fallback: untrained ViT tiny (predicts something, low confidence).
        from algorithms.vit import ViTPredictor
        return ViTPredictor()

    def _weather_backbone(self, m: str):
        mod = _import_mod(m)
        pred_cls = mod.__dict__[{"vit": "ViTPredictor",
                                 "swin": "SwinPredictor",
                                 "convnext": "ConvNeXtPredictor",
                                 "efficientformer": "EfficientFormerPredictor"}[m]]
        return pred_cls(weights_path=self._best_ckpt(m))

    def _objects(self):
        from algorithms.yolo import YOLODetector
        return YOLODetector()

    def _lanes(self):
        from algorithms.lane_detection import LaneDetector
        return LaneDetector(mode="classical")

    def _signs(self):
        from algorithms.traffic_sign import TrafficSignPredictor
        return TrafficSignPredictor(weights_path=self._best_ckpt("traffic_sign"))

    def _seg(self):
        from algorithms.segmentation import SegmentationPredictor
        return SegmentationPredictor()


def _import_mod(name: str):
    return __import__(f"algorithms.{name}", fromlist=[name])


REGISTRY = _Registry()


# ======================================================================= #
#  High-level helpers used by the API layer
# ======================================================================= #
def get(task: str):
    """Return a model instance for `task`, or None if it failed to build."""
    return REGISTRY.build(task)


def build_status() -> dict:
    """Health summary of every task (built / error / not attempted)."""
    out = {"tasks": {}, "errors": {}, "models_dir": str(MODELS_DIR)}
    for t in ("weather", "weather_models", "objects", "lanes", "signs", "seg"):
        state = "ready" if t in REGISTRY._cache else (
            "error" if t in REGISTRY._errors else "lazy")
        out["tasks"][t] = state
        if t in REGISTRY._errors:
            out["errors"][t] = REGISTRY._errors[t]
    # checkpoint availability
    out["checkpoints"] = {m: bool(self_ckpt(m)) for m in BACKBONES}
    out["backbones"] = BACKBONES
    return out


def self_ckpt(m: str):
    return (MODELS_DIR / m / "best.pt").exists()


def load_comparison() -> list:
    """Read results/comparison.json -> list of rows (or [])."""
    p = RESULTS_DIR / "comparison.json"
    if not p.exists():
        return []
    import json
    return json.loads(p.read_text())


def analyze_image(bgr, tasks=None, save_overlay_path=None):
    """Run the selected perception tasks on one BGR frame.

    tasks : iterable of keys (default all). Returns
        {
          "success": bool,
          "tasks":   {key: <JSON contract or error>},
          "overlay": {
              "saved_path": str|None,
              "draw":      {objects, lanes, seg, signs, weather, ...} rendering info
          },
          "inference_time_ms": float,
        }
    """
    import time
    import numpy as np

    tasks = list(tasks or ["weather", "objects", "lanes", "signs", "seg"])
    out = {"success": True, "tasks": {}, "inference_time_ms": 0.0}
    t0 = time.perf_counter()

    for t in tasks:
        model = get(t)
        if model is None:
            out["tasks"][t] = {"status": "error",
                               "error": f"model unavailable ({REGISTRY._errors.get(t)})"}
            continue
        try:
            out["tasks"][t] = model.predict(bgr)
        except Exception as e:
            traceback.print_exc()
            out["tasks"][t] = {"status": "error", "error": str(e)}

    out["inference_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # ------------------------------------------------------------- overlay
    overlay_bgr = bgr.copy()
    info = {"objects": 0, "lanes": 0, "seg_classes": 0, "sign": None,
            "weather": None, "weather_conf": None}

    try:
        # objects (YOLO boxes)
        objs = out["tasks"].get("objects", {}).get("results", {}).get("detections", []) or []
        for o in objs[:20]:
            try:
                x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
                cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay_bgr, f"{o['class']} {o['confidence']*100:.1f}%",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                pass
        info["objects"] = len(objs)

        # lanes
        from algorithms.lane_detection.classical import draw_lanes
        lanes = out["tasks"].get("lanes", {}).get("results", {}).get("lanes", []) or []
        if lanes:
            overlay_bgr = draw_lanes(overlay_bgr, lanes)
        info["lanes"] = len(lanes)

        # segmentation: blend colour mask over the frame
        seg = out["tasks"].get("seg")
        if seg and seg.get("status") == "success":
            seg_mod = get("seg")
            mask_p = seg["results"].get("mask_path")
            if mask_p and seg_mod is not None and _P(mask_p).exists():
                mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)
                if mask is not None and mask.shape[:2] == overlay_bgr.shape[:2]:
                    overlay_bgr = seg_mod._colourise(mask, overlay_bgr, alpha=0.45)
            info["seg_classes"] = seg["results"].get("num_classes", 0)

        # notice text box
        weather = out["tasks"].get("weather", {})
        info["weather"] = weather.get("prediction")
        info["weather_conf"] = weather.get("confidence")
        signs = out["tasks"].get("signs", {})
        info["sign"] = signs.get("prediction")

        _draw_textbar(overlay_bgr, info, out.get("inference_time_ms"))
    except Exception as e:
        traceback.print_exc()

    # optional save
    saved = None
    if save_overlay_path:
        saved = str(save_overlay_path)
        _P(save_overlay_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay_path), overlay_bgr)
    out["overlay"] = {"saved_path": saved, "info": info}

    # helper to hand back base64 without saving
    out["overlay_bgr"] = overlay_bgr
    return out


def _draw_textbar(bgr, info, ms):
    h, w = bgr.shape[:2]
    lines = [
        f"WX: {info['weather']} {0 if info['weather_conf'] is None else round(info['weather_conf']*100)}%",
        f"objs:{info['objects']}  lanes:{info['lanes']}  seg:{info['seg_classes']}cl",
        f"sign: {info['sign']}",
        f"ADAS {round(1000.0/max(ms,0.001))} FPS",
    ]
    cv2.rectangle(bgr, (0, 0), (int(w*0.5), 84), (0, 0, 0), -1)
    for i, l in enumerate(lines):
        cv2.putText(bgr, l, (10, 22 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)


def process_video(path, tasks=None, max_frames=None, out_video=None,
                  sample_every=1):
    """Run analyze_image over a video, write an annotated MP4."""
    import cv2
    import time

    tasks = list(tasks or ["weather", "objects", "lanes", "signs", "seg"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_video or (RESULTS_DIR / "integration" /
                             f"analysis_{Path(path).stem}_{int(time.time())}.mp4")
    _P(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    frames_done = 0
    per_timings = []
    first_frame = None            # 1 annotated sample frame for the UI
    stats = {"frames": 0, "weather_counts": {}, "object_frames": 0, "avg_ms": 0.0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frames_done >= max_frames:
            break
        t0 = time.perf_counter()
        r = analyze_image(frame, tasks=tasks)
        annotated = r["overlay_bgr"]
        dt = (time.perf_counter() - t0) * 1000.0
        per_timings.append(dt)
        writer.write(annotated)
        frames_done += 1
        wx = r["overlay"]["info"]["weather"]
        stats["weather_counts"][str(wx)] = stats["weather_counts"].get(str(wx), 0) + 1
        if r["overlay"]["info"]["objects"]:
            stats["object_frames"] += 1
        if first_frame is None:
            first_frame = annotated.copy()
        cap.read if False else None
    cap.release()
    writer.release()

    stats["frames"] = frames_done
    stats["avg_ms"] = round(sum(per_timings) / max(len(per_timings), 1), 2)
    stats["min_ms"], stats["max_ms"] = (round(min(per_timings), 2),
                                        round(max(per_timings), 2))
    stats["video_path"] = str(out_path)
    return stats, first_frame
