"""
bdd_to_yolo.py - optional helper to convert BDD100K detection labels
into Ultralytics YOLO format for fine-tuning.

Use this ONLY if you want to fine-tune YOLO on BDD detection classes
(10 categories: pedestrian, rider, car, ...).

Reads BDD100K 2018 train/val labels plus images, writes:

  datasets/bdd100k_yolo/
    images/{train,val}/<name>.jpg     (copies)
    labels/{train,val}/<name>.txt     (YOLO labels: cls cx cy w h, normalized)
    bdd100k.yaml                      (YOLO dataset card)

Run:
    python algorithms/yolo/bdd_to_yolo.py
    python algorithms/yolo/train.py --data datasets/bdd100k_yolo/bdd100k.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

CONFIDENCE_FIXED = True  # BDD2018 doesn't carry detection conf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(CONFIDENCE_FIXED):
    None  # noqa
BDD_LABELS_TO_IDX = {
    "pedestrian":    0, "rider":     1, "car":   2, "truck":  3, "bus":   4,
    "train":         5, "motorcycle": 6, "bicycle": 7, "traffic light": 8,
    "traffic sign":  9, "other person": 0, "other vehicle": 2, "trailer": 3,
}

# We normalize the few BDD aliases to the canonical 10 classes
BDD_ALIAS = {
    "other person": "pedestrian", "person": "pedestrian",
    "other vehicle": "car", "trailer": "truck",
}


def to_yolo_line(box, cls_idx, w, h):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw =  (x2 - x1) / w
    bh =  (y2 - y1) / h
    return f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def convert_split(label_json: Path, images_in: Path, images_out: Path, labels_out: Path):
    with open(label_json) as f:
        frames = json.load(f)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    n_objects = 0
    for frame in frames:
        name = frame["name"]
        src  = images_in / name
        if not src.exists():
            continue
        # image dimensions are 1280x720 per BDD spec
        W, H = 1280, 720
        lines = []
        for lab in frame.get("labels", []):
            cat = BDD_ALIAS.get(lab["category"], lab["category"])
            if cat not in BDD_LABELS_TO_IDX:
                continue
            box = lab["box2d"]
            lines.append(to_yolo_line(box, BDD_LABELS_TO_IDX[cat], W, H))
            n_objects += 1
        dst_img  = images_out / name
        dst_lab  = labels_out  / (Path(name).stem + ".txt")
        # only copy if we have at least one detection
        if lines:
            shutil.copy2(src, dst_img)
            dst_lab.write_text("\n".join(lines))
    return len(list(images_out.glob("*"))), n_objects


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bdd-root", type=Path, default=PROJECT_ROOT / "datasets" / "bdd100k")
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "datasets" / "bdd100k_yolo")
    p.add_argument("--max-images", type=int, default=None,
                   help="cap per split (quick experiments); default = all")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, lab_file, img_dir in [
        ("train", args.bdd_root/ "labels" / "bdd100k_labels_images_train.json",
                  args.bdd_root/ "images" / "100k" / "train"),
        ("val",   args.bdd_root/ "labels" / "bdd100k_labels_images_val.json",
                  args.bdd_root/ "images" / "100k" / "val"),
    ]:
        n_imgs, n_objs = convert_split(
            lab_file, img_dir,
            args.out_root / "images" / split,
            args.out_root / "labels" / split,
        )
        summary[split] = {"images": n_imgs, "objects": n_objs}
        print(f"{split}: {n_imgs} images, {n_objs} objects converted")

    # write the yaml dataset card
    yaml_path = args.out_root / "bdd100k.yaml"
    yaml_path.write_text(
        f"path: {args.out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names: {[k for k,_ in sorted(BDD_LABELS_TO_IDX.items(), key=lambda x: x[1])]}\n"
    )
    print(f"done. yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
