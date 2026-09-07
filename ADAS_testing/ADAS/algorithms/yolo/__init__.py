"""Algorithm 6 - YOLOv8 object detection.

Public API:
    YOLODetector(...).predict(image) -> dict

Uses ultralytics. yolov8n weights (=~6 MB) are auto-downloaded by ultralytics
on first construct unless you pass `weights=` and the file already exists.
"""

from .model import YOLODetector, BDD_TO_YOLO_CLASSES, ROAD_RELEVANT_COCO_CLASSES

__all__ = ["YOLODetector", "BDD_TO_YOLO_CLASSES", "ROAD_RELEVANT_COCO_CLASSES"]
