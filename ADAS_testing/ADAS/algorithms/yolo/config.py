"""YOLOv8 module configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_OUT    = MODELS_DIR / "yolo"
RESULTS_OUT  = RESULTS_DIR / "yolo"

# Default pretrained checkpoint (COCO 80 classes)
DEFAULT_WEIGHTS = "yolov8n.pt"

# Use COCO-trained (out-of-box) or BDD100K-finetuned (custom)
DEFAULT_TASK = "detect"

# Confidence threshold for predictions
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# Which COCO classes we consider "road-relevant" for our ADAS context
# (subset of all 80 COCO classes)
ROAD_CLASSES = {
    0: "person",       # pedestrian
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",   # partial traffic-sign coverage
}

# BDD100K detection has 10 classes (see bdd_to_yolo.py)
BDD_CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle", "traffic light", "traffic sign"]
