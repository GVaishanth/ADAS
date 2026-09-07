# Algorithm 7 - Lane Detection

Two modes, same `predict(image) -> dict` contract:

## 1. `classical` (default, recommended for the prototype)
- OpenCV Canny + Hough + ROI mask + left/right slope split + polynomial fit.
- Works out of the box: no model file, no training.
- Detection quality depends on contrast of lane markings - works well on
  well-marked highways at day.

## 2. `dl`
- Lightweight U-Net (`UNetLane`, ~250k params) trained on **TuSimple** to
  predict a 2-channel mask (left lane, right lane).
- Requires you to run `train.py` on TuSimple to produce the checkpoint.
- Note: TuSimple is a highway-lane benchmark (3 lanes wide, constrained
  scenarios). For a general ADAS, prefer the classical detector plus a
  separate deep-learning detector if you need fine-grained curves.

## Files
- `model.py`         LaneDetector class (both modes).
- `classical.py`     OpenCV pure-image lane pipeline (reused for viz).
- `tusimple.py`      TuSimple dataset loader (optional).
- `train.py`         DL mode training scaffold.
- `predict.py`       Image / video / webcam inference.

## Routes
| Action | Command |
|---|---|
| Image (classical) | `python algorithms/lane_detection/predict.py --image <path>.jpg` |
| Webcam (classical) | `python algorithms/lane_detection/predict.py --source 0` |
| Train U-Net on TuSimple | `python algorithms/lane_detection/train.py --tusimple-root datasets/tusimple --epochs 10` |
| Use DL mode | `python algorithms/lane_detection/predict.py --image ... --mode dl --weights models/lane_detection/lane_unet.pt` |

## TuSimple dataset (only for DL mode)
- Source: https://github.com/TuSimple/tusimple-benchmark
- Expected layout:
      datasets/tusimple/
          label_data_*.json     (train/val/test labels)
          clips/<clip>/20.jpg    (image frames)
- Size: ~5 GB.
- 3 lanes per frame, ~3600 training images / ~900 val.

## API contract
```json
{
  "status": "success",
  "prediction": "lanes_2",
  "confidence": 0.6,
  "inference_time_ms": 8.4,
  "results": {
    "num_lanes": 2,
    "lanes": [
      {"slope": -0.40, "intercept": 540, "points": [[x,y], ...]},
      {"slope":  0.31, "intercept": 520, "points": [[x,y], ...]}
    ],
    "overlay_path": "results/lane_detection/.../lane.png",
    "mode": "classical"
  }
}
```
