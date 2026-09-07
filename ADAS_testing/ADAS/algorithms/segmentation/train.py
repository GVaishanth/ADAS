"""
train.py - fine-tune SegFormer on Cityscapes (or ACDC).

This is a SCAFFOLD. To use it:

  1. Get Cityscapes (requires registration):
         https://www.cityscapes-dataset.com/downloads
     Unzip a few files into:
         datasets/cityscapes/leftImg8bit/{train,val}/<city>/<file>_leftImg8bit.png
         datasets/cityscapes/gtFine/{train,val}/<city>/<file>_gtFine_labelTrainIds.png
     (Train IDs are produced by the official `createTrainIdLabelImgs.py`)
  2. Run:
         python algorithms/segmentation/train.py
"""
import argparse, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cityscapes-root", type=Path, default=config.CITYSCAPES_DIR)
    p.add_argument("--epochs",   type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--batch",    type=int, default=config.DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=config.DEFAULT_LR)
    args = p.parse_args()
    pth = Path(args.cityscapes_root)
    if not (pth / "leftImg8bit" / "train").exists():
        raise SystemExit(f"Cityscapes not found at {pth}. See module README.")
    print("TODO: full fine-tune script. Use this as a starting point.")
    print("Tip: install torchmetrics for mIoU.")
    # Avoid actually doing fine-tuning in sandbox: just confirm modules import.
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(config.HF_MODEL_ID)
    print(f"Loaded pretrained SegFormer ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params).")

if __name__ == "__main__":
    main()
