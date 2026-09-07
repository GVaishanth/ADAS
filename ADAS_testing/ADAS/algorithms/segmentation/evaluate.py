"""
evaluate.py - compute mIoU / pixel accuracy on the Cityscapes val split
with the pretrained SegFormer that the predictor already loads.

Requires Cityscapes val labels (`*_gtFine_labelTrainIds.png`).

Outputs:  results/segmentation/metrics.json  + confusion-matrix PNG
"""

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .model import CITYSCAPES_CLASSES


def compute_mIoU(confusion):
    hist = confusion.astype(np.float64)
    IoUs = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)
    return float(np.nanmean(IoUs)), {cls: float(v) for cls, v in zip(CITYSCAPES_CLASSES, IoUs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cityscapes-root", type=Path, default=config.CITYSCAPES_DIR)
    p.add_argument("--img-size", type=int, default=config.DEFAULT_IMG_SIZE)
    p.add_argument("--max-images", type=int, default=50, help="limit val images for quick eval")
    args = p.parse_args()

    from PIL import Image as PILImage
    from transformers import SegformerImageProcessor

    img_root = args.cityscapes_root / "leftImg8bit" / "val"
    lbl_root = args.cityscapes_root / "gtFine"    / "val"
    if not img_root.exists():
        raise SystemExit(f"Cityscapes val missing at {img_root}.")

    from .model import nn_resize_logits
    from transformers import SegformerForSemanticSegmentation

    device = config.get_device()
    model = SegformerForSemanticSegmentation.from_pretrained(config.HF_MODEL_ID).to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(config.HF_MODEL_ID)

    # gather val pairs
    pairs = []
    for city in sorted(img_root.iterdir()):
        for f in sorted(city.glob("*_leftImg8bit.png")):
            city_part = f.parent.name
            base = f.stem.replace("_leftImg8bit", "")
            lbl_path = lbl_root / city_part / f"{base}_gtFine_labelTrainIds.png"
            if lbl_path.exists():
                pairs.append((f, lbl_path))
            if len(pairs) >= args.max_images:
                break
        if len(pairs) >= args.max_images:
            break

    # pixel confusion matrix
    n_cls = len(CITYSCAPES_CLASSES)
    confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
    pix_correct, pix_total = 0, 0
    for img_p, lbl_p in pairs:
        bgr = __import__("cv2").imread(str(img_p))
        rgb = __import__("cv2").cvtColor(bgr, __import__("cv2").COLOR_BGR2RGB)
        gt = np.array(PILImage.open(lbl_p))   # uint8 with trainIds 0..18
        inputs = processor(images=PILImage.fromarray(rgb), return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        # resize prediction to gt size
        pred_full = __import__("cv2").resize(pred, (gt.shape[1], gt.shape[0]),
                                              interpolation=__import__("cv2").INTER_NEAREST)
        # ignore label 255 / out-of-range
        mask = (gt >= 0) & (gt < n_cls)
        confusion[np.ix_(gt[mask], pred_full[mask])] += 1
        pix_correct += int((pred_full[mask] == gt[mask]).sum())
        pix_total   += int(mask.sum())

    miou, ious = compute_mIoU(confusion)
    pix_acc = pix_correct / max(pix_total, 1)

    metrics = {
        "model": config.HF_MODEL_ID,
        "split": "val",
        "images_used": len(pairs),
        "mIoU": miou,
        "class_IoUs": ious,
        "pixel_accuracy": pix_acc,
        "note": "TBD until run on real val images",
    }
    config.RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(config.RESULTS_OUT / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
