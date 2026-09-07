"""
transformer_base.py
====================
Base classes used by every timm-based classifier
(ViT, Swin, ConvNeXt, EfficientFormer).

The four comparative models differ ONLY in their timm model name, so this
module gives them a single base implementation. Each algorithm module then
provides a 10-line subclass pointing at its own model name.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from algorithms.preprocessing.preprocessing import get_val_transforms
from algorithms.preprocessing.config import WEATHER_CLASSES


class TimmClassifier(nn.Module):
    """The model used during training (raw nn.Module)."""

    def __init__(self, num_classes: int, pretrained: bool, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)


class TimmPredictor:
    """JSON-friendly predictor wrapping a TimmClassifier."""

    def __init__(self, weights_path=None, num_classes=6, model_name="",
                 device=None, class_names=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or WEATHER_CLASSES

        # IMPORTANT: no pretrained download here. Caller controls whether
        # the underlying classifier was trained.
        self.model = TimmClassifier(num_classes=num_classes, pretrained=False,
                                     model_name=self.model_name)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        self.transform = get_val_transforms()

    @torch.no_grad()
    def predict(self, image) -> dict:
        """Standard FastAPI / webcam JSON contract."""
        t0 = time.perf_counter()
        bgr = self._load(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_idx = int(probs.argmax())
        dt = (time.perf_counter() - t0) * 1000.0
        top_k = sorted(
            [{"class": self.class_names[i], "confidence": float(probs[i])}
             for i in range(len(self.class_names))],
            key=lambda r: -r["confidence"],
        )[:3]
        return {
            "status": "success",
            "prediction": self.class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "inference_time_ms": round(dt, 3),
            "results": {
                "top_k": top_k,
                "logits_dim": int(self.num_classes),
                "model": self.model_name,
            },
        }

    @staticmethod
    def _load(image):
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"could not read image: {image}")
            return bgr
        return image
