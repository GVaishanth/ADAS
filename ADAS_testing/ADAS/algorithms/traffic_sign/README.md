# Algorithm 8 - Traffic Sign Recognition

43-class GTSRB classifier. Backbone: timm `resnet18`.

## Files
- `model.py`     TrafficSignClassifier + TrafficSignPredictor.
- `dataset.py`   GTSRB image-folder loader.
- `train.py`     Fine-tune / train from scratch.
- `evaluate.py`  Test metrics + confusion matrix.
- `predict.py`   Image / video / webcam inference.

## Dataset
- Source: https://benchmark.ini.rub.de/gtsrb_dataset.html (free).
- Expected folder layout (after unzipping):
      datasets/gtsrb/
          Training/<class_id>/<file>.ppm
          Testing/<class_id>/<file>.ppm
- Class count: **43** German traffic signs.
- Storage: ~300 MB.
- Alternative: pass `--auto-download` to use `torchvision.datasets.GTSRB(...)`.

## Routes
| Action | Command |
|---|---|
| Image | `python algorithms/traffic_sign/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/traffic_sign/predict.py --source 0` |
| Train (with download) | `python algorithms/traffic_sign/train.py --auto-download --epochs 8` |
| Evaluate | `python algorithms/traffic_sign/evaluate.py --weights models/traffic_sign/best.pt` |

## API contract
```json
{
  "status": "success",
  "prediction": "Speed limit (30km/h)",
  "confidence": 0.94,
  "inference_time_ms": 8.1,
  "results": {
    "class_id": 1,
    "top_k": [
      {"class_id": 1, "class_name": "Speed limit (30km/h)", "confidence": 0.94},
      {"class_id": 2, "class_name": "Speed limit (50km/h)", "confidence": 0.03},
      ...
    ],
    "model": "resnet18",
    "img_size": 64
  }
}
```
