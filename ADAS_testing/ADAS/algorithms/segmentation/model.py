"""
model.py - Semantic segmentation predictor backed by HuggingFace SegFormer.

Pretrained model: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
downloaded (~15 MB) on first construct. Trained on Cityscapes 19 classes.
For ACDC-style fine-tuning on adverse weather, point at the ACDC
fine-tune script (TODO scaffold in train.py) after downloading ACDC.

Predictor returns the FastAPI JSON contract plus:
  - results.mask_path    : uint8 PNG where each pixel value = class_id
  - results.overlay_path : visualisation (BGR) blended with the input
  - results.classes      : list of class_names actually present in this frame
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config


class SegmentationPredictor:
    """JSON-friendly SegFormer wrapper."""

    def __init__(self, model_id: str = None, device=None,
                 input_size: int = None):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        # lazy heavy imports
        self.model_id = model_id or config.HF_MODEL_ID
        self.device = device or config.get_device()
        self.input_size = input_size or config.DEFAULT_IMG_SIZE
        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = (SegformerForSemanticSegmentation
                          .from_pretrained(self.model_id)
                          .to(self.device).eval())
        self.class_names = config.CITYSCAPES_CLASSES
        self.colormap    = config.CITYSCAPES_COLORMAP

    @torch.no_grad()
    def predict(self, image, save_overlay_path=None, save_mask_path=None) -> dict:
        from PIL import Image
        t0 = time.perf_counter()
        bgr = self._load(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # segmenter expects list of PIL images
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits  = outputs.logits                  # [1, C, h, w]
        # nearest-neighbour resize back to processor size, then to image size
        upsampled = nn_resize_logits(logits, shapes=tuple(inputs["pixel_values"].shape[2:]))[0]
        seg_map   = upsampled.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # resize mask back to the original image resolution
        original_size = bgr.shape[:2]
        seg_full = cv2.resize(seg_map, (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_NEAREST)

        # stats
        classes, counts = np.unique(seg_full, return_counts=True)
        total = int(seg_full.size)
        present = [{"class_id": int(c), "class_name": self.class_names[int(c)]
                      if 0 <= int(c) < len(self.class_names) else "unknown",
                    "pixel_count": int(n), "percent": round(100 * n / total, 2)}
                   for c, n in zip(classes, counts)]

        # save mask + overlay
        overlay_full = None
        overlay_path = None; mask_path = None
        out_dir = Path(config.RESULTS_OUT)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_mask_path is None and save_overlay_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
        if save_mask_path is not None:
            mask_path = str(save_mask_path); cv2.imwrite(mask_path, seg_full)
        if save_overlay_path is not None:
            overlay_full = self._colourise(seg_full, bgr, alpha=0.55)
            overlay_path = str(save_overlay_path); cv2.imwrite(overlay_path, overlay_full)

        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "success",
            "prediction": f"scene_with_{len(present)}_classes",
            "confidence": round(float(1.0 / max(len(present), 1)), 4),
            "inference_time_ms": round(dt, 3),
            "results": {
                "num_classes":   len(present),
                "classes":       present,
                "mask_path":     mask_path,
                "overlay_path":  overlay_path,
                "model":         self.model_id,
                "input_size":    self.input_size,
                "note":          "mIoU requires ground-truth masks - not computed here",
            },
        }

    # ---------------- helpers ----------------
    def _colourise(self, mask, bgr, alpha=0.5):
        """Return an BGR image with colour-mapped mask blended over the input."""
        colour = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id, col in enumerate(self.colormap):
            colour[mask == cls_id] = col       # BGR
        return cv2.addWeighted(bgr, 1 - alpha, colour, alpha, 0)

    @staticmethod
    def _load(image):
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"could not read image: {image}")
            return bgr
        return image


def nn_resize_logits(logits, shapes):
    """Helper for the newer transformers API.
    Resizes (1, C, h, w) logits to the requested spatial size."""
    if hasattr(logits, "logits"):
        logits = logits.logits
    if logits.dim() == 4 and logits.shape[:2] != shapes[:1] + (logits.shape[1],):
        # interpolate
        import torch.nn.functional as F
        logits = F.interpolate(logits, size=shapes, mode="bilinear", align_corners=False)
    return logits
