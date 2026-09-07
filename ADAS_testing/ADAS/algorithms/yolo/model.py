"""
model.py - YOLODetector wrapper around ultralytics.YOLO.

Constructor pulls weights if missing (yolov8n.pt ~= 6 MB). The predict()
method follows the FastAPI JSON contract used by every other algorithm.
"""

import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from . import config

# Re-export helpers
BDD_TO_YOLO_CLASSES = config.BDD_CLASSES
ROAD_RELEVANT_COCO_CLASSES = config.ROAD_CLASSES


class YOLODetector:
    def __init__(self, weights: Union[str, Path] = None, conf: float = None,
                 iou: float = None, device: str = "auto",
                 road_only: bool = True):
        """road_only=True filters out non-road COCO classes so the dashboard
        sees only road-relevant objects (car, person, traffic light, ...)."""
        from ultralytics import YOLO         # heavy import, lazy

        self.weights = str(weights) if weights else config.DEFAULT_WEIGHTS
        self.conf = conf if conf is not None else config.CONF_THRESHOLD
        self.iou  = iou  if iou  is not None else config.IOU_THRESHOLD
        self.road_only = road_only

        # ultralytics handles "auto" -> picks GPU if available
        self.model = YOLO(self.weights)
        if device != "auto":
            self.model.to(device)

        # Read class names from the model meta
        self.class_names = self.model.names
        self._road_idx   = set(config.ROAD_CLASSES.keys())

    def predict(self, image, save_annotated_path: Union[str, Path] = None) -> dict:
        """image: path or BGR ndarray. Returns the standard FastAPI JSON dict."""
        t0 = time.perf_counter()
        # ultralytics accepts both path and ndarray
        results = self.model.predict(source=image,
                                      conf=self.conf, iou=self.iou,
                                      verbose=False)
        if not results:
            return {"status": "success", "prediction": None, "confidence": None,
                    "inference_time_ms": 0.0, "results": {"detections": []}}
        r = results[0]
        boxes = r.boxes
        detections = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                if self.road_only and int(k) not in self._road_idx:
                    continue
                name = self.class_names[int(k)]
                detections.append({
                    "class": name,
                    "confidence": round(float(c), 4),
                    "bbox": [round(float(p), 2) for p in (x1, y1, x2, y2)],
                    "class_id": int(k),
                })

        # optional annotated save
        if save_annotated_path is not None:
            try:
                annotated = r.plot()           # BGR numpy
                cv2.imwrite(str(save_annotated_path), annotated)
                annotated_path = str(save_annotated_path)
            except Exception:
                annotated_path = None
        else:
            annotated_path = None

        if not detections:
            top_conf = 0.0; top_class = None
        else:
            top = max(detections, key=lambda d: d["confidence"])
            top_conf = top["confidence"]; top_class = top["class"]

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "success",
            "prediction": top_class,
            "confidence": top_conf,
            "inference_time_ms": round(dt, 3),
            "results": {
                "num_detections": len(detections),
                "detections":     detections,
                "annotated_path": annotated_path,
                "weights":        self.weights,
                "conf_threshold": self.conf,
            },
        }
