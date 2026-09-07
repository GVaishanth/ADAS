"""
tusimple.py - minimal TuSimple dataset loader.

TuSimple format:
    one JSON file `label_data_*.json` per split (train/val/test).
    each entry is a dict with:  keys = list of (h_sample, y) for each lane;
    raw_file -> JPEG path (under <tusimple>/clips/.../20.jpg).

This loader returns ONE binary segmentation mask per lane (the most useful
form for training a U-Net / SegFormer-style lane detector).  For the prototype
the DL mode in model.py uses a small U-Net that takes the camera frame and
predicts (left_lane_mask, right_lane_mask) - we won't ship the DL training
end-to-end, only the scaffold.

The CLASSICAL mode (default) does NOT need TuSimple at all.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# TuSimple images are 1280x720, lanes are list of (x, y) coordinates
IMG_H, IMG_W = 720, 1280


def _polyline_to_mask(label, img_path, target_size=(96, 160)):
    """Turn one TuSimple frame's annotations into a 2-channel mask.
    Channel 0 = left lane, channel 1 = right lane (rough heuristic:
    left = points whose average x < image-width / 2)."""
    mask = np.zeros((IMG_H, IMG_W, 2), dtype=np.uint8)
    h_samples = label["h_samples"]
    lanes     = label["lanes"]
    for pts in lanes:
        if len(pts) != len(h_samples):
            continue
        xs = [x for x, valid in zip(pts, h_samples) if valid != -2]
        ys = [y for y, valid in zip(h_samples, pts)        if valid != -2]
        if not xs:
            continue
        avg_x = float(np.mean(xs))
        ch = 0 if avg_x < IMG_W / 2 else 1
        for x, y in zip(xs, ys):
            if 0 <= x < IMG_W and 0 <= y < IMG_H:
                mask[int(y), int(x), ch] = 255
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    mask_small = cv2.resize(mask, (target_size[1], target_size[0]))
    return img, mask_small


class TuSimpleDataset(Dataset):
    """Returns (image_tensor [3,H,W], mask_tensor [2,H,W]).

    Set target_size smaller than full image for a prototype, e.g. (96, 160).
    """
    def __init__(self, label_json: Path, clips_root: Path,
                 transform=None, target_size=(96, 160)):
        self.entries = []
        with open(label_json) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(json.loads(line))
        self.clips_root  = clips_root
        self.target_size = target_size
        self.tx = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(target_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tx = lambda m: torch.from_numpy(m).permute(2, 0, 1).float() / 255.0

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        label = self.entries[idx]
        img_path = self.clips_root / label["raw_file"]
        image, mask = _polyline_to_mask(label, img_path, self.target_size)
        if image is None:
            # very rare; returning a tiny black frame + black mask keeps it sane
            image = np.zeros((IMG_H, IMG_W, 3), np.uint8)
            mask  = np.zeros((self.target_size[0], self.target_size[1], 2), np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.tx(image), self.mask_tx(mask)
