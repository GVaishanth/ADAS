"""
model.py - Traffic-sign classifier.

Backbone: timm resnet18 (small enough to train from scratch on a laptop/GPU).
Predictor returns the standard FastAPI JSON contract plus a `top_k` list of
the 5 highest-scoring classes.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from . import config


class TrafficSignClassifier(nn.Module):
    """Replaces timm's classifier head with an Identity+Linear -> num_classes."""

    def __init__(self, num_classes: int = 43, pretrained: bool = False,
                 model_name: str = None):
        super().__init__()
        self.model_name = model_name or config.TIMM_MODEL
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


class TrafficSignPredictor:
    """JSON-friendly traffic-sign predictor."""

    def __init__(self, weights_path=None, num_classes=43, model_name=None,
                 device=None, class_names=None):
        self.model_name = model_name or config.TIMM_MODEL
        self.num_classes = num_classes
        self.device = device or config.get_device()
        self.class_names = class_names or config.GTSRB_CLASS_NAMES
        self.model = TrafficSignClassifier(num_classes=num_classes, pretrained=False,
                                            model_name=self.model_name)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        mean = [0.3403, 0.3121, 0.3214]
        std  = [0.2724, 0.2608, 0.2669]
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @torch.no_grad()
    def predict(self, image) -> dict:
        t0 = time.perf_counter()
        bgr = self._load(image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_idx = int(probs.argmax())
        dt = (time.perf_counter() - t0) * 1000.0
        top_k = sorted(
            [{"class_id": int(i),
              "class_name": self.class_names[int(i)],
              "confidence": float(probs[int(i)])}
             for i in np.argsort(probs)[::-1][:5]],
            key=lambda r: -r["confidence"],
        )
        return {
            "status": "success",
            "prediction": self.class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "inference_time_ms": round(dt, 3),
            "results": {
                "class_id":  top_idx,
                "top_k":     top_k,
                "model":     self.model_name,
                "img_size":  config.IMG_SIZE,
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


# Export class names for convenience
GTSRB_CLASS_NAMES = config.GTSRB_CLASS_NAMES
