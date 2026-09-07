# Algorithm 2 - Vision Transformer (ViT)

Road-scene weather classification, comparative model #1.

## What this module does
- Fine-tunes timm's `vit_base_patch16_224.augreg2_in21k_ft_in1k` on the
  BDD100K 6-class weather subset built by Algorithm 1.
- Exposes `ViTPredictor.predict(image) -> dict` for FastAPI / webcam.

## Files
- `model.py`        ViTClassifier (nn.Module) + ViTPredictor (JSON-friendly).
- `train.py`        Fine-tune on the train manifest; saves best by val acc.
- `evaluate.py`     Test-set evaluation + confusion matrix + append to
                    shared `results/comparison.json`.
- `predict.py`      Image / video / webcam inference with annotation.

## Routes
| Action | Command |
|---|---|
| Train | `python algorithms/vit/train.py --pretrained --epochs 5` |
| Evaluate | `python algorithms/vit/evaluate.py` |
| Image | `python algorithms/vit/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/vit/predict.py --source 0` |

## Inputs / outputs
- Same manifest CSVs as Algorithms 3, 4, 5 (`datasets/bdd100k_weather/`).
- Same transforms as Algorithm 1 (`get_train_transforms()`, `get_val_transforms()`).
- Saves weights to `models/vit/best.pt` and metrics to `results/vit/metrics.json`.

## Notes
- `--pretrained` downloads ~330 MB from timm on first run.
- The first command line runs on the **same** random seed and split
  counts used by Swin / ConvNeXt / EfficientFormer.
- This module never fabricates metrics; only runs on real test data
  will populate the comparison row.
