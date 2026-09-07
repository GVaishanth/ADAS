"""
build_dataset.py
================
One-time dataset builder for the weather-condition classification task.

What it does
------------
1. Reads BDD100K label JSONs (train + val) and collects, per weather class,
   the image file names.
2. Drops "undefined" weather images and missing files (counts reported).
3. Builds a stratified prototype subset:
       train : sampled from the official BDD100K train split
       val   : sampled from the official BDD100K val split
       test  : sampled from the REMAINING official-val images
               (BDD100K test labels are not public, so our test set
                comes from val and is guaranteed disjoint from our val set)
4. Writes one manifest CSV per split + class_names.json + dataset_report.json.
5. Optionally copies images into a classic ImageFolder layout
   (output/images/<split>/<class>/...) if --copy-images is used.
6. Saves a class-distribution chart to results/preprocessing/.

Usage
-----
    python algorithms/preprocessing/build_dataset.py
    python algorithms/preprocessing/build_dataset.py --copy-images --train-per-class 50

Run it from the ADAS/ project root (paths are absolute, so any cwd works).
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

# Allow running this file directly from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing import config


def collect_by_weather(label_json: Path, images_dir: Path):
    """Read one BDD100K label file -> {weather: [absolute image paths]}.

    Also returns counts of images skipped because weather == "undefined"
    or because the file is missing on disk.
    """
    with open(label_json, "r", encoding="utf-8") as f:
        frames = json.load(f)          # list of frame dicts

    per_class, undefined, missing = {}, 0, 0
    for frame in frames:
        name = frame.get("name")
        weather = frame.get("attributes", {}).get("weather", "undefined")

        if weather == "undefined":
            undefined += 1
            continue
        img_path = images_dir / name
        if not img_path.exists():
            missing += 1
            continue
        per_class.setdefault(weather, []).append(str(img_path.resolve()))

    return per_class, undefined, missing


def stratified_sample(per_class: dict, n: int, rng: random.Random) -> dict:
    """Pick up to n random paths per class. If a class has fewer than n
    samples, everything available is used (and reported)."""
    chosen = {}
    for cls, paths in per_class.items():
        chosen[cls] = rng.sample(paths, min(n, len(paths)))
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build the BDD100K weather-condition subset")
    parser.add_argument("--train-per-class", type=int, default=config.TRAIN_PER_CLASS,
                        help="train images per weather class")
    parser.add_argument("--val-per-class", type=int, default=config.VAL_PER_CLASS,
                        help="val images per weather class")
    parser.add_argument("--test-per-class", type=int, default=config.TEST_PER_CLASS,
                        help="test images per weather class")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed")
    parser.add_argument("--copy-images", action="store_true",
                        help="also copy images into an ImageFolder layout "
                             "(output/images/<split>/<class>/...)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ------------------------------------------------------------------ #
    # 1. Read the two official label files
    # ------------------------------------------------------------------ #
    print("Reading BDD100K labels ...")
    train_all, undef_tr, miss_tr = collect_by_weather(config.BDD_LABELS_TRAIN,
                                                      config.BDD_SPLIT_IMAGES["train"])
    val_all, undef_val, miss_val = collect_by_weather(config.BDD_LABELS_VAL,
                                                      config.BDD_SPLIT_IMAGES["val"])
    print(f"  train labels parsed: {sum(len(v) for v in train_all.values()):,} images "
          f"({undef_tr} undefined, {miss_tr} missing files)")
    print(f"  val   labels parsed: {sum(len(v) for v in val_all.values()):,} images "
          f"({undef_val} undefined, {miss_val} missing files)")

    # ------------------------------------------------------------------ #
    # 2. Stratified subsetting
    # ------------------------------------------------------------------ #
    train_chosen = stratified_sample(train_all, args.train_per_class, rng)
    val_chosen   = stratified_sample(val_all, args.val_per_class, rng)

    # test set: from the val images NOT already used by our val set
    test_chosen = {}
    for cls, paths in val_all.items():
        used = set(val_chosen.get(cls, []))
        remaining = [p for p in paths if p not in used]
        test_chosen[cls] = rng.sample(remaining, min(args.test_per_class, len(remaining)))

    # ------------------------------------------------------------------ #
    # 3. Write manifest CSVs
    # ------------------------------------------------------------------ #
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, chosen in [("train", train_chosen),
                          ("val", val_chosen),
                          ("test", test_chosen)]:
        for cls, paths in chosen.items():
            for p in paths:
                rows.append({"image_path": p, "label": cls, "split": split})

    df = pd.DataFrame(rows)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split][["image_path", "label"]]
        sub.to_csv(config.OUTPUT_DIR / f"manifest_{split}.csv", index=False)
        print(f"  wrote manifest_{split}.csv  ({len(sub)} rows)")

    with open(config.OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(config.WEATHER_CLASSES, f, indent=2)

    # ------------------------------------------------------------------ #
    # 4. Report + optional copy + chart
    # ------------------------------------------------------------------ #
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = df[df["split"] == split]["label"].value_counts().to_dict()

    report = {
        "seed": args.seed,
        "classes": config.WEATHER_CLASSES,
        "counts_per_split": counts,
        "skipped_undefined": {"train": undef_tr, "val": undef_val},
        "skipped_missing_files": {"train": miss_tr, "val": miss_val},
        "full_train_weather_distribution": {k: len(v) for k, v in train_all.items()},
        "full_val_weather_distribution": {k: len(v) for k, v in val_all.items()},
    }
    with open(config.OUTPUT_DIR / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.copy_images:
        print("Copying images into ImageFolder layout ...")
        for split, chosen in [("train", train_chosen),
                              ("val", val_chosen),
                              ("test", test_chosen)]:
            for cls, paths in chosen.items():
                dst_dir = config.OUTPUT_DIR / "images" / split / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    shutil.copy2(p, dst_dir / Path(p).name)

    # class-distribution chart -> results/preprocessing/
    config.RESULTS_PREP.mkdir(parents=True, exist_ok=True)
    plot_df = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    ax = plot_df.plot(kind="bar", figsize=(10, 5), rot=0,
                      title="Prototype weather subset - images per class & split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.legend(title="Weather")
    plt.tight_layout()
    chart_path = config.RESULTS_PREP / "class_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  chart saved -> {chart_path}")

    # summary table
    print("\n=== Prototype subset summary ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    print(f"\nAll manifests written under: {config.OUTPUT_DIR}")
    print("Next step: python algorithms/preprocessing/evaluate.py")


if __name__ == "__main__":
    main()
