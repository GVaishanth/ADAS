"""
make_synthetic_bdd.py  (TEST ONLY)
==================================
Generates a tiny FAKE BDD100K (images + label JSONs) so you can verify the
whole Algorithm-1 pipeline BEFORE downloading the real ~7 GB dataset.

Creates under datasets/bdd100k/:
    images/100k/train/*.jpg   (8 per weather class)
    images/100k/val/*.jpg     (5 per weather class)
    labels/bdd100k_labels_images_train.json
    labels/bdd100k_labels_images_val.json

Each fake image is a small 160x90 picture with a colour cast per weather
class (blue = rainy, grey = foggy, ...) so the previews are visually distinct.

Usage:
    python datasets/make_synthetic_bdd.py
    python algorithms/preprocessing/build_dataset.py --train-per-class 5 \
        --val-per-class 3 --test-per-class 2
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the ADAS/ folder
BDD = PROJECT_ROOT / "datasets" / "bdd100k"

W, H = 160, 90
# Per-weather (sky, road) colour pairs so each class has a distinct cast
# while keeping photo-like brightness/contrast (see note below).
SCENES = {
    "clear":         ((210, 215, 225), (70, 75, 80)),
    "partly cloudy": ((200, 205, 215), (75, 80, 85)),
    "overcast":      ((175, 180, 190), (70, 75, 80)),
    "rainy":         ((120, 140, 175), (55, 60, 70)),
    "snowy":         ((235, 238, 242), (150, 155, 160)),
    "foggy":         ((185, 190, 195), (95, 100, 105)),
}
PER_CLASS_TRAIN, PER_CLASS_VAL = 8, 5


def fake_image(weather: str, rng: random.Random) -> np.ndarray:
    """Small pseudo-photo with photo-like statistics.

    We deliberately include bright sky + dark road + many small high-contrast
    details + noise, so the per-channel variance after ImageNet normalization
    lands near std ~ 1 (like a real photograph) instead of ~0.3 (flat colour).
    """
    rnd = np.random.default_rng(rng.randint(0, 999))
    sky, road = SCENES[weather]

    # gradient sky (top 60%) fading toward the road colour
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        for c in range(3):
            img[y, :, c] = int(sky[c] + (road[c] - sky[c]) * max(0.0, min(1.0, t / 0.6)))

    # road + lane marking
    cv2.rectangle(img, (0, int(H * 0.6)), (W, H), road, -1)
    cv2.line(img, (int(W * 0.5), int(H * 0.6)), (int(W * 0.5), H), (200, 210, 60), 3)

    # many small bright AND dark details -> local contrast like a real scene
    for _ in range(18):
        x1, y1 = rng.randint(0, W - 10), rng.randint(0, H - 6)
        x2, y2 = min(x1 + rng.randint(3, 9), W), min(y1 + rng.randint(2, 6), H)
        bright = rng.randint(0, 1) == 0
        col = tuple(int(c) for c in (rnd.integers(200, 255, 3) if bright
                                     else rnd.integers(0, 55, 3)))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)

    # headlight highlight + white road dash
    cv2.circle(img, (int(W * 0.8), int(H * 0.3)), 8, (250, 250, 240), -1)
    cv2.circle(img, (int(W * 0.2), int(H * 0.75)), 5, (255, 255, 255), -1)

    # sensor noise (rain/snow grain)
    noise = rnd.integers(-18, 19, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def write_split(split: str, n_per_class: int, label_path: Path):
    rng = random.Random(100 + (0 if split == "train" else 1))
    img_dir = BDD / "images" / "100k" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for weather in SCENES:
        for i in range(n_per_class):
            name = f"{weather.replace(' ', '_')}_{i:02d}_{split}.jpg"
            cv2.imwrite(str(img_dir / name), fake_image(weather, rng))
            frames.append({
                "name": name,
                "attributes": {"weather": weather,
                               "scene": "city street",
                               "timeofday": "daytime"},
                "labels": [],
            })
    # extra frames to exercise the "undefined" and "missing file" counters
    frames.append({"name": "no_weather_00.jpg",
                   "attributes": {"weather": "undefined"},
                   "labels": []})
    frames.append({"name": "ghost_file.jpg",
                   "attributes": {"weather": "rainy"},
                   "labels": []})
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        json.dump(frames, f, indent=1)
    print(f"wrote {label_path}  ({len(frames)} frames, {n_per_class} images/class)")


def main():
    write_split("train", PER_CLASS_TRAIN,
                BDD / "labels" / "bdd100k_labels_images_train.json")
    write_split("val", PER_CLASS_VAL,
                BDD / "labels" / "bdd100k_labels_images_val.json")
    print("Synthetic BDD100K ready. Now run build_dataset.py with small sizes.")


if __name__ == "__main__":
    main()
