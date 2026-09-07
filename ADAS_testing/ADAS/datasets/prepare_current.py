"""
prepare_current.py
==================
Adapts the CURRENT provided dataset (the four folders that ship in
``ADAS/datasets/{Day,Night,Fog,Rain}``) into the manifest-CSV layout that the
whole ML pipeline (Algorithms 1-9) consumes.

Why this file exists
--------------------
The repo's original ``build_dataset.py`` expects a **BDD100K** download
(``datasets/bdd100k/images/100k/...`` + the two label JSONs). The dataset we
actually have is a small set of gathered dashcam frames, one folder per driving
condition:

    datasets/Day    -> "clear"
    datasets/Night  -> "night"
    datasets/Fog    -> "foggy"
    datasets/Rain   -> "rainy"

This script copies/points those folders at the standard manifest layout so
*every* downstream module (vit / swin / convnext / efficientformer / yolo /
lane / traffic_sign / segmentation / webcam integration) works unchanged.

Output (under ``datasets/bdd100k_weather/``):
    manifest_train.csv, manifest_val.csv, manifest_test.csv
    class_names.json, dataset_report.json, class_distribution.png
    images/<split>/<class>/....jpg      (only with --copy-images)

Usage:
    python datasets/prepare_current.py                # uses ALL images (80/10/10 split)
    python datasets/prepare_current.py --val-ratio 0.15 --test-ratio 0.10
    python datasets/prepare_current.py --val-per-class 25 --test-per-class 25
    python datasets/prepare_current.py --copy-images
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # the ADAS/ folder
DATA_ROOT    = PROJECT_ROOT / "datasets"

# Folder name -> our condition class label (single source of truth).
FOLDER_TO_CLASS = {
    "Day":   "clear",
    "Night": "night",
    "Fog":   "foggy",
    "Rain":  "rainy",
}

# The 4 classes of the current dataset, in a stable order.
CURRENT_CLASSES = ["clear", "night", "foggy", "rainy"]


# Names that are junk sidecars / metadata, never a real image: macOS AppleDouble
# (._foo.png, ._foo.jpg), hidden files, .DS_Store, Thumbs.db, desktop.ini, etc.
_JUNK_SUFFIX_MARKERS = ("._", "thumbs.db", ".ds_store", "desktop.ini", "@eaDir")


def _is_real_image(p: Path) -> bool:
    """True only for a real, loadable image; False for junk sidecars / corrupt files.

    Filters out macOS AppleDouble files (``._photo.png``), hidden/system metadata
    (``.DS_Store``, ``Thumbs.db``, ``desktop.ini``) and picks up genuine .jpg/.jpeg/.png.
    It additionally opens each candidate so a truncated/corrupt image can never be
    written into the manifest and crash training later get a bad path.
    """
    name = p.name.lower()
    if not p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return False
    if name.startswith("._") or name.startswith("."):
        return False
    if any(m in name for m in _JUNK_SUFFIX_MARKERS if m != "._"):
        return False
    # Final safety net: verify the bytes are actually a decodable image.
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        print(f"      !! dropping unreadable image: {p}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the current {Day,Night,Fog,Rain} dataset for the pipeline")
    # By default we USE ALL images. Override either the ratios or, if you prefer,
    # a fixed count per class. If a per-class count is given it wins over the ratio.
    parser.add_argument("--train-per-class", type=int, default=None,
                        help="fixed # of training images per class (default: all remaining after val/test)")
    parser.add_argument("--val-per-class", type=int, default=None,
                        help="fixed # of val images per class (default: round(n * --val-ratio))")
    parser.add_argument("--test-per-class", type=int, default=None,
                        help="fixed # of test images per class (default: round(n * --test-ratio))")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="fraction reserved for validation (default 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="fraction reserved for test (default 0.10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true",
                        help="copy images into an ImageFolder layout as well")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------ #
    # 1. Collect every image per class
    # ------------------------------------------------------------------ #
    per_class = {}
    for folder, cls in FOLDER_TO_CLASS.items():
        src = DATA_ROOT / folder
        if not src.exists():
            print(f"  !! skip {folder}: folder not found at {src}")
            continue
        paths = sorted(str(p.resolve()) for p in src.glob("*")
                       if _is_real_image(p))
        per_class[cls] = paths
        print(f"  {cls:6s} <- {folder}/ : {len(paths)} images")

    if not per_class:
        raise SystemExit("No dataset folder found. Check ADAS/datasets/.")

    # ------------------------------------------------------------------ #
    # 2. Stratified split per class (USES EVERY image). val/test default to
    #    a fraction of each class; train gets everything left over.
    # ------------------------------------------------------------------ #
    chosen = {"train": {}, "val": {}, "test": {}}
    for cls, paths in per_class.items():
        rng.shuffle(paths)
        n = len(paths)
        # per-class counts win over ratios when provided; otherwise use ratios
        n_te = args.test_per_class if args.test_per_class is not None \
            else int(round(n * args.test_ratio))
        n_va = args.val_per_class if args.val_per_class is not None \
            else int(round(n * args.val_ratio))
        # never let val/test exceed what's available, and keep >=1 for tiny classes
        n_te = max(1, min(n_te, max(n - 1, 0)))
        n_va = max(1, min(n_va, max(n - n_te, 0)))
        n_tr = n - n_te - n_va                    # train gets ALL the rest
        chosen["train"][cls] = paths[:n_tr]
        chosen["val"][cls]   = paths[n_tr:n_tr + n_va]
        chosen["test"][cls]  = paths[n_tr + n_va:n_tr + n_va + n_te]

    # ------------------------------------------------------------------ #
    # 3. Write manifests + class_names.json
    # ------------------------------------------------------------------ #
    out_dir = DATA_ROOT / "bdd100k_weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "val", "test"):
        for cls, paths in chosen[split].items():
            for p in paths:
                rows.append({"image_path": p, "label": cls, "split": split})
    df = pd.DataFrame(rows)

    for split in ("train", "val", "test"):
        sub = df[df["split"] == split][["image_path", "label"]]
        sub.to_csv(out_dir / f"manifest_{split}.csv", index=False)
        print(f"  wrote manifest_{split}.csv  ({len(sub)} rows)")

    (out_dir / "class_names.json").write_text(json.dumps(CURRENT_CLASSES, indent=2))

    # ------------------------------------------------------------------ #
    # 4. Report + chart + optional image copy
    # ------------------------------------------------------------------ #
    counts = {s: df[df["split"] == s]["label"].value_counts().to_dict()
              for s in ("train", "val", "test")}
    report = {
        "source": "current {Day,Night,Fog,Rain} folders",
        "classes": CURRENT_CLASSES,
        "counts_per_split": counts,
        "total_images_used": int(len(df)),
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))

    if args.copy_images:
        print("  copying images into ImageFolder layout ...")
        for split in ("train", "val", "test"):
            for cls, paths in chosen[split].items():
                dst_dir = out_dir / "images" / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    shutil.copy2(p, dst_dir / Path(p).name)

    results_prep = PROJECT_ROOT / "results" / "preprocessing"
    results_prep.mkdir(parents=True, exist_ok=True)
    plot_df = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0,
                      title="Current dataset - images per class & split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.legend(title="Condition")
    plt.tight_layout()
    chart = results_prep / "class_distribution.png"
    plt.savefig(chart, dpi=150)
    plt.close()
    print(f"  chart saved -> {chart}")

    print("\n=== Summary ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nAll manifests written under: {out_dir}")
    print("Next: train + evaluate the four transformer backbones "
          "(algorithms/{vit,swin,convnext,efficientformer}/).")


if __name__ == "__main__":
    main()
