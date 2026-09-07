# Algorithm 5 - EfficientFormer

Road-scene weather classification, comparative model #4 (latency-friendly).

## Files
- `model.py`     EfficientFormerClassifier + EfficientFormerPredictor.
- `train.py`     Fine-tune efficientformer_l1.snap_dist_in1k.
- `evaluate.py`  Test split + append row to comparison.json.
- `predict.py`   Image / video / webcam.

## Routes
| Action | Command |
|---|---|
| Train | `python algorithms/efficientformer/train.py --pretrained --epochs 5` |
| Evaluate | `python algorithms/efficientformer/evaluate.py` |
| Image | `python algorithms/efficientformer/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/efficientformer/predict.py --source 0` |

## Notes
- Uses the same BDD manifest + same transforms as the other three models.
- EfficientFormer has ~12M params, latency-friendly for the laptop target.
