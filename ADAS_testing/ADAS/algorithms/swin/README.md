# Algorithm 3 - Swin Transformer

Road-scene weather classification, comparative model #2.

## Files
- `model.py`     SwinClassifier + SwinPredictor (5-line subclass of shared base).
- `train.py`     Fine-tune swin_tiny_patch4_window7_224.
- `evaluate.py`  Test split + append row to comparison.json.
- `predict.py`   Image / video / webcam inference.

## Routes
| Action | Command |
|---|---|
| Train | `python algorithms/swin/train.py --pretrained --epochs 5` |
| Evaluate | `python algorithms/swin/evaluate.py` |
| Image | `python algorithms/swin/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/swin/predict.py --source 0` |

## Notes
- Uses the same BDD manifest + same transforms as ViT, ConvNeXt, EfficientFormer.
- `--pretrained` downloads ~110 MB.
- Model size: Swin-Tiny ~28M params, ~110 MB on disk.
