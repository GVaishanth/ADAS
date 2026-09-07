# Algorithm 6 - YOLOv8 Object Detection

Road-object detection using Ultralytics YOLOv8.

## Out-of-box (no training)
- Uses `yolov8n.pt` (~6 MB, downloaded on first run).
- Already detects: car, truck, bus, motorcycle, bicycle, person, **traffic light**, **stop sign**.
- Configure with `--all-classes` to keep every COCO class (80), or leave the default for road-only filtering.

## Files
- `model.py`       YOLODetector (JSON-friendly wrapper).
- `bdd_to_yolo.py` Converts BDD100K detection labels to Ultralytics YOLO format (optional).
- `train.py`       Fine-tunes YOLO on BDD100K detection subset.
- `evaluate.py`    Computes mAP/Precision/Recall on the BDD-YOLO val set.
- `predict.py`     Image / video / webcam inference.

## Routes
| Action | Command |
|---|---|
| Image (COCO) | `python algorithms/yolo/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/yolo/predict.py --source 0` |
| Convert BDD to YOLO | `python algorithms/yolo/bdd_to_yolo.py` |
| Fine-tune | `python algorithms/yolo/train.py --data datasets/bdd100k_yolo/bdd100k.yaml` |
| Evaluate | `python algorithms/yolo/evaluate.py --weights ... --data ...` |

## Dataset
- For out-of-box use: **no dataset needed** (COCO weights).
- For fine-tuning:
  - `bdd_to_yolo.py` reads `bdd100k/labels/bdd100k_labels_images_*.json` and the
    `images/100k/{train,val}` jpg folder, builds `datasets/bdd100k_yolo/`.
  - 10 classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign.
  - Storage: ~5 GB if you copy images with `--max-images`; up to ~7 GB for full.

## API contract
```json
{
  "status": "success",
  "prediction": "person",          # top-confidence detected class
  "confidence":   0.91,
  "inference_time_ms": 22.4,
  "results": {
    "num_detections": 3,
    "detections": [
       {"class": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2], "class_id": 0},
       ...
    ],
    "annotated_path": "results/yolo/<file>_yolo.png",
    "weights":        "yolov8n.pt",
    "conf_threshold": 0.25
  }
}
```
