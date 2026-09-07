# Algorithm 4 - ConvNeXt

Road-scene weather classification, comparative model #3 (modernized ConvNet).

## Files
- `model.py`     ConvNeXtClassifier + ConvNeXtPredictor.
- `train.py`     Fine-tune convnext_tiny.fb_in22k_ft_in1k.
- `evaluate.py`  Test split + append row to comparison.json.
- `predict.py`   Image / video / webcam.

## Routes
| Action | Command |
|---|---|
| Train | `python algorithms/convnext/train.py --pretrained --epochs 5` |
| Evaluate | `python algorithms/convnext/evaluate.py` |
| Image | `python algorithms/convnext/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/convnext/predict.py --source 0` |

## Notes
- Uses the same BDD manifest + same transforms as ViT, Swin, EfficientFormer.
- ConvNeXt is a CNN-baseline in the comparison (modernized, design fits ViT-style training).
