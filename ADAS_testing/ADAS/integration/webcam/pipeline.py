"""
pipeline.py - WebcamPipeline.

A single class that opens the laptop webcam (or any video source) and runs
the full ADAS chain each frame:

    webcam bgr frame
        |
        v
    preprocessing  (optional)
        |
        v
    ---------- AI models ----------------------------------
        |
        v
    weather classifier (ViT/Swin/ConvNeXt/EfficientFormer)
    object detector (YOLOv8)
    lane detector (classical OR dl)
    traffic sign (GTSRB)
    semantic segmentation (SegFormer)
        |
        v
    ---------- aggregator -------------------------------
        |
        v
    JSON-friendly dict for FastAPI / frontend dashboard

Pass `model_instances` to inject heavy models you've already built; otherwise
default lightweight models are constructed.

JSON shape returned by `step(frame)`:
    {
      "timestamp": 1734567890.123,
      "frame_id": 42,
      "inference_time_ms": 75.3,
      "weather": {...},         # full weather predictor result
      "objects": [...],         # YOLO detections
      "lanes":   [...],         # lane polyline points
      "signs":   {...},         # traffic sign predictor result
      "segments":{...},         # segmentation result (counts only - mask not embedded)
    }
"""

import json
import sys
import time
import traceback
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent        # ADAS/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing.preprocessing import PreprocessingPipeline
from . import config


class WebcamPipeline:
    """Real-time multi-model ADAS pipeline."""

    def __init__(self, weather_pred=None, object_det=None, lane_det=None,
                 sign_pred=None, seg_pred=None, preprocess_pipe=None,
                 enable_segmentation=True, enable_signs=True,
                 enable_object_detection=True, enable_lanes=True,
                 enable_weather=True):
        # Models (defaults: cheaper / non-pre-trained where possible)
        self.weather = weather_pred
        self.objects = object_det
        self.lanes   = lane_det
        self.signs   = sign_pred
        self.segments = seg_pred
        self.preprocess = preprocess_pipe or PreprocessingPipeline()

        self.enable_segmentation      = enable_segmentation and (seg_pred is not None)
        self.enable_signs             = enable_signs and (sign_pred is not None)
        self.enable_object_detection  = enable_object_detection and (object_det is not None)
        self.enable_lanes             = enable_lanes and (lane_det is not None)
        self.enable_weather           = enable_weather and (weather_pred is not None)
        self._frame_id = 0

    # --------------------------------------------------------------
    #  one-frame step
    # --------------------------------------------------------------
    def step(self, bgr_frame) -> dict:
        out = {
            "frame_id": self._frame_id,
            "timestamp": round(time.time(), 3),
        }
        t0 = time.perf_counter()
        results = {}
        # weather classification
        if self.enable_weather and self.weather is not None:
            try:
                results["weather"] = self.weather.predict(bgr_frame)
            except Exception as e:
                results["weather"] = {"status": "error", "error": str(e)}
        # object detection
        if self.enable_object_detection and self.objects is not None:
            try:
                results["objects"] = self.objects.predict(bgr_frame)
            except Exception as e:
                results["objects"] = {"status": "error", "error": str(e)}
        # lane detection
        if self.enable_lanes and self.lanes is not None:
            try:
                results["lanes"] = self.lanes.predict(bgr_frame)
            except Exception as e:
                results["lanes"] = {"status": "error", "error": str(e)}
        # traffic sign
        if self.enable_signs and self.signs is not None:
            try:
                results["signs"] = self.signs.predict(bgr_frame)
            except Exception as e:
                results["signs"] = {"status": "error", "error": str(e)}
        # segmentation (just class counts, not the full mask)
        if self.enable_segmentation and self.segments is not None:
            try:
                seg = self.segments.predict(bgr_frame)
                results["segments"] = {
                    "num_classes":  seg["results"].get("num_classes"),
                    "classes":      [c["class_name"] for c in seg["results"].get("classes", [])],
                    "top_class":    (sorted(seg["results"].get("classes", []),
                                            key=lambda c: -c["percent"])[0] if seg["results"].get("classes") else None),
                    "inference_time_ms": seg.get("inference_time_ms"),
                }
            except Exception as e:
                results["segments"] = {"status": "error", "error": str(e)}

        out.update(results)
        out["inference_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self._frame_id += 1
        return out

    # --------------------------------------------------------------
    #  long-running loop
    # --------------------------------------------------------------
    def run(self, source=0, max_frames=None, display=True,
            save_video_path: str = None):
        """Open `source` (int -> webcam, str -> video path), run step() on every
        frame, optional viewport + optional annotated video write."""
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise SystemExit(f"could not open source: {source}")
        writer = None; n_frames = 0; total_ms = 0
        is_webcam = str(source).isdigit()
        stub = (source if not is_webcam else f"webcam_{int(time.time())}")
        out_path = Path(save_video_path) if save_video_path else (
            config.RESULTS_OUT / f"webcam_run_{stub}.mp4")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"running on {source} ... press ESC to stop (webcam).")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.perf_counter()
                result = self.step(frame)
                dt = (time.perf_counter() - t0) * 1000.0
                total_ms += dt; n_frames += 1

                # simple dashboard overlay (bottom third of the frame)
                annotated = frame
                if display or save_video_path:
                    annotated = self._draw_overlay(frame, result)

                if writer is None and annotated is not None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(out_path),
                                              cv2.VideoWriter_fourcc(*"mp4v"),
                                              cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
                if writer is not None:
                    writer.write(annotated)

                if display and is_webcam:
                    cv2.imshow("ADAS Webcam", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                if max_frames is not None and n_frames >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows() if display else None

        avg = total_ms / max(n_frames, 1)
        print(f"sessions: {self._frame_id} frames | "
              f"{n_frames} processed | avg {avg:.1f} ms | {1000.0/max(avg,1):.1f} FPS | "
              f"output -> {out_path}")
        return out_path

    # --------------------------------------------------------------
    #  lightweight visual overlay
    # --------------------------------------------------------------
    def _draw_overlay(self, bgr, result):
        out = bgr.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (8, 8), (340, 120), (0, 0, 0), -1)

        # weather
        w_pred = result.get("weather", {}).get("prediction", "?")
        conf   = result.get("weather", {}).get("confidence", 0.0)
        cv2.putText(out, f"WX: {w_pred}  {conf*100:.1f}%", (14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        # objects
        objs = result.get("objects", {}).get("results", {}).get("detections", []) or []
        cv2.putText(out, f"OBJ: {len(objs)}", (14, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        # lanes
        lns = result.get("lanes", {}).get("results", {}).get("num_lanes", 0)
        cv2.putText(out, f"LANE: {lns}", (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        # signs
        s_pred = result.get("signs", {}).get("prediction", "?") or "?"
        cv2.putText(out, f"SIGN: {s_pred[:20]}", (14, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        # FPS
        cv2.putText(out, f"ADAS {1000.0/max(result['inference_time_ms'],1):.1f} FPS",
                    (14, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # YOLO bboxes from result if present
        for obj in objs:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{obj['class']} {obj['confidence']*100:.1f}%",
                        (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Lane lines
        from algorithms.lane_detection.classical import draw_lanes
        lane_data = result.get("lanes", {}).get("results", {}).get("lanes", [])
        if lane_data:
            out = draw_lanes(out, lane_data)
        return out
