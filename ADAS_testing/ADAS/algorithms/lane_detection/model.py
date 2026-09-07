"""
model.py - LaneDetector class.

mode='classical'   OpenCV-based, works out of the box, no model file.
mode='dl'          Small UNetLane (lightweight; created on demand).
                   Returns the SAME JSON shape for both modes.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .classical import detect_lanes, draw_lanes
from . import config


class UNetLane(torch.nn.Module):
    """Tiny U-Net (encoder ~250k params) for binary left/right lane masks.
    Used only when mode='dl' and the user has produced weights via train.py."""

    def __init__(self, n_classes=2):
        super().__init__()
        def block(ic, oc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ic, oc, 3, padding=1), torch.nn.BatchNorm2d(oc), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(oc, oc, 3, padding=1), torch.nn.BatchNorm2d(oc), torch.nn.ReLU(inplace=True),
            )
        self.d0 = block(3, 32); self.p0 = torch.nn.MaxPool2d(2)
        self.d1 = block(32, 64); self.p1 = torch.nn.MaxPool2d(2)
        self.bottleneck = block(64, 128)
        self.u1  = torch.nn.ConvTranspose2d(128, 64, 2, stride=2); self.d2 = block(128, 64)
        self.u0  = torch.nn.ConvTranspose2d(64,  32, 2, stride=2); self.d3 = block(64,  32)
        self.head = torch.nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x0 = self.d0(x)
        x  = self.p0(x0)
        x1 = self.d1(x)
        x  = self.p1(x1)
        x  = self.bottleneck(x)
        x  = self.d2(torch.cat([self.u1(x), x1], dim=1))
        x  = self.d3(torch.cat([self.u0(x), x0], dim=1))
        return torch.sigmoid(self.head(x))


class LaneDetector:
    """JSON-friendly lane detector.

    Usage:
        det = LaneDetector(mode='classical')          # no weights needed
        res = det.predict('image.jpg',                # saves annotated overlay
                          save_overlay_path='results/lane/overlay.png')
    """

    def __init__(self, mode: str = "classical", weights: (str, Path) = None, device=None):
        self.mode = mode
        self.device = device or config.get_device()
        if self.mode == "dl":
            self.model = UNetLane().to(self.device).eval()
            if weights and Path(weights).exists():
                state = torch.load(weights, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
        else:
            self.model = None

    def predict(self, image, save_overlay_path=None) -> dict:
        t0 = time.perf_counter()
        bgr = self._load(image)
        if self.mode == "classical":
            lanes, _ = detect_lanes(bgr)
        else:
            lanes = self._dl_predict(bgr)

        overlay_path = None
        if save_overlay_path is not None:
            save_overlay_path = Path(save_overlay_path)
            save_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = draw_lanes(bgr, lanes)
            cv2.imwrite(str(save_overlay_path), annotated)
            overlay_path = str(save_overlay_path)

        total_pts = sum(len(l["points"]) for l in lanes)
        confidence = min(1.0, total_pts / (config.NUM_PTS_PER_LANE * 2))
        dt = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "success",
            "prediction": f"lanes_{len(lanes)}",
            "confidence": round(float(confidence), 4),
            "inference_time_ms": round(dt, 3),
            "results": {
                "num_lanes":     len(lanes),
                "lanes":         [{"slope": l["slope"], "intercept": l["intercept"],
                                    "points": l["points"]} for l in lanes],
                "overlay_path":  overlay_path,
                "mode":          self.mode,
            },
        }

    @torch.no_grad()
    def _dl_predict(self, bgr):
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((96, 160)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = tf(rgb).unsqueeze(0).to(self.device)
        mask = self.model(x).squeeze(0).cpu().permute(1, 2, 0).numpy()
        # crude lane extraction per channel -> find non-empty columns top-bottom -> sample 20 points
        out = []
        H, W, _ = mask.shape
        target_h, target_w = bgr.shape[:2]
        for ch in range(mask.shape[2]):
            m = mask[:, :, ch]
            ys, xs = np.where(m > 0.5)
            if len(xs) < 5:
                continue
            slope, intercept = np.polyfit(ys, xs, 1)
            ys_eval = np.linspace(int(H * config.DEFAULT_ROI_VB_RATIO[0]),
                                   int(H * config.DEFAULT_ROI_VB_RATIO[1]),
                                   config.NUM_PTS_PER_LANE)
            xs_eval = (slope * ys_eval + intercept).astype(int)
            ys_e = (ys_eval * target_h / H).astype(int)
            xs_e = (xs_eval * target_w / W).astype(int)
            out.append({"slope": float(slope), "intercept": float(intercept),
                         "points": [[int(xx), int(yy)] for xx, yy in zip(xs_e, ys_e)]})
        return out

    @staticmethod
    def _load(image):
        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"could not read image: {image}")
            return bgr
        return image
