"""
evaluate.py
===========
Validates the preprocessing pipeline and the built dataset.

Checks
------
1. Manifests exist for train / val / test.
2. Splits are disjoint (no image appears in two splits).
3. Every listed image file exists and a sample opens without corruption.
4. Per-class counts + imbalance ratio.
5. Timing benchmark: average ms per preprocessed image and baseline FPS.
6. Tensor sanity: output shape [3,224,224], float32, mean ~ 0 and std ~ 1
   after ImageNet normalization.

Outputs
-------
    results/preprocessing/evaluation_report.json   (machine-readable)
    results/preprocessing/pipeline_preview.png     (original vs enhanced vs model-input grid)
    console table
"""

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.preprocessing import config
from algorithms.preprocessing.preprocessing import (
    PreprocessingPipeline,
    apply_enhancement,
    tensor_to_image,
)


def check_file_integrity(sample_size=30):
    """Every listed file must exist; a random sample must open cleanly."""
    issues = []
    for split in ("train", "val", "test"):
        csv_path = config.OUTPUT_DIR / f"manifest_{split}.csv"
        if not csv_path.exists():
            issues.append(f"missing manifest_{split}.csv (run build_dataset.py first)")
            continue
        df = pd.read_csv(csv_path)
        missing = [p for p in df["image_path"] if not Path(p).exists()]
        if missing:
            issues.append(f"{split}: {len(missing)} missing files, e.g. {missing[0]}")
        sample = df["image_path"].sample(min(sample_size, len(df)), random_state=config.SEED)
        corrupt = [p for p in sample if cv2.imread(str(p)) is None]
        if corrupt:
            issues.append(f"{split}: {len(corrupt)} corrupt/could-not-open sample files")
    return issues


def disjointness_report():
    sets = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(config.OUTPUT_DIR / f"manifest_{split}.csv")
        sets[split] = set(df["image_path"])
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlaps[f"{a}-{b}"] = len(sets[a] & sets[b])
    return overlaps


def timing_benchmark(num_images=100):
    """Average ms per image + baseline FPS on the laptop CPU pipeline."""
    df = pd.read_csv(config.OUTPUT_DIR / "manifest_val.csv")
    df = df.sample(min(num_images, len(df)), random_state=config.SEED)
    pipe = PreprocessingPipeline(enhance="none")
    times, shapes = [], []
    for p in df["image_path"]:
        t0 = time.perf_counter()
        out = pipe.preprocess(p)
        times.append(out["inference_time_ms"])
        shapes.append(out["results"]["tensor_shape"])
    avg_ms = float(pd.Series(times).mean())
    return {
        "images_tested": len(times),
        "avg_ms_per_image": round(avg_ms, 3),
        "std_ms": round(float(pd.Series(times).std()), 3),
        "min_ms": round(float(min(times)), 3),
        "max_ms": round(float(max(times)), 3),
        "baseline_fps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0,
        "output_shape": shapes[0] if shapes else None,
    }


def tensor_sanity(pipe):
    """Deterministic checks that do NOT depend on dataset content:

    1. A constant-gray image (128,128,128) must map to exactly
       (128/255 - mean[c]) / std[c] per channel. This catches a forgotten or
       wrongly-parameterised Normalize step, whatever the data looks like.
    2. Output shape [3, 224, 224] and dtype float32.

    Real-photo aggregate mean/std are returned as INFORMATION ONLY (adverse-
    weather colour casts shift them, so they are not a pass/fail signal).
    """
    const = np.full((config.INPUT_SIZE, config.INPUT_SIZE, 3), 128, dtype=np.uint8)
    t = pipe.preprocess_tensor(const)

    expected = [(128.0 / 255.0 - m) / s
                for m, s in zip(config.IMAGENET_MEAN, config.IMAGENET_STD)]
    actual = [round(float(v), 4) for v in t.mean(dim=[1, 2])]

    shape_ok = list(t.shape) == [3, config.INPUT_SIZE, config.INPUT_SIZE]
    dtype_ok = str(t.dtype) == "torch.float32"
    checks = {
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "constant_image_expected": [round(e, 4) for e in expected],
        "constant_image_actual": actual,
        "normalization_ok": (shape_ok and dtype_ok and
                             all(abs(a - e) < 0.01 for a, e in zip(actual, expected))),
    }

    # informational aggregate stats on a small photo sample
    df = pd.read_csv(config.OUTPUT_DIR / "manifest_val.csv")
    df = df.sample(min(20, len(df)), random_state=0)
    means, stds = [], []
    for p in df["image_path"]:
        tt = pipe.preprocess_tensor(p)
        means.append(tt.mean(dim=[1, 2]).tolist())
        stds.append(tt.std(dim=[1, 2]).tolist())
    checks["photo_avg_channel_mean"] = [
        round(float(sum(m[i] for m in means) / len(means)), 3) for i in range(3)]
    checks["photo_avg_channel_std"] = [
        round(float(sum(s[i] for s in stds) / len(stds)), 3) for i in range(3)]
    return checks


def preview_grid(out_path: Path):
    """Original | CLAHE enhanced | model-input grid, one row per weather class."""
    df = pd.read_csv(config.OUTPUT_DIR / "manifest_val.csv")
    pipe = PreprocessingPipeline(enhance="clahe")

    n_rows = len(config.WEATHER_CLASSES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 2.4 * n_rows))
    for row, cls in enumerate(config.WEATHER_CLASSES):
        sub = df[df["label"] == cls]
        if sub.empty:
            for col in range(3):
                axes[row, col].axis("off")
            continue
        p = sub.iloc[0]["image_path"]
        bgr = cv2.imread(str(p))
        enhanced, _ = apply_enhancement(bgr, "clahe")
        tensor = pipe.preprocess_tensor(bgr)
        model_view = tensor_to_image(tensor)

        axes[row, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"Original\n{cls}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title("CLAHE enhanced", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(cv2.cvtColor(model_view, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title("Model input\n(224x224, normalized)", fontsize=9)
        axes[row, 2].axis("off")

    plt.suptitle("Preprocessing pipeline - one sample per weather class", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=== Algorithm 1 - Preprocessing evaluation ===\n")
    config.RESULTS_PREP.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. dataset-level checks
    print("[1/4] Dataset / file-integrity checks ...")
    issues = check_file_integrity()
    overlaps = disjointness_report()
    report["file_integrity_issues"] = issues
    report["split_overlaps"] = overlaps
    for pair, n in overlaps.items():
        print(f"      overlap {pair}: {n} images")
    if issues:
        print("      PROBLEMS found:")
        for i in issues:
            print(f"        - {i}")
    else:
        print("      OK - all manifests fine, no missing/corrupt sample files")

    # 2. class balance
    print("[2/4] Class balance (train split) ...")
    df = pd.read_csv(config.OUTPUT_DIR / "manifest_train.csv")
    counts = df["label"].value_counts()
    report["class_balance"] = {
        "train_counts": counts.to_dict(),
        "imbalance_ratio_max_min": round(float(counts.max() / max(counts.min(), 1)), 2),
    }
    print(counts.to_string())

    # 3. timing benchmark
    print("[3/4] Preprocessing timing benchmark ...")
    timing = timing_benchmark()
    report["timing"] = timing
    print(f"      avg {timing['avg_ms_per_image']} ms/image -> "
          f"{timing['baseline_fps']} FPS baseline (laptop CPU, no enhancement)")

    # 4. tensor sanity
    print("[4/4] Tensor sanity (deterministic constant-image check) ...")
    sanity = tensor_sanity(PreprocessingPipeline())
    report["tensor_sanity"] = sanity
    print(f"      constant 128-gray image -> expected {sanity['constant_image_expected']}, "
          f"got {sanity['constant_image_actual']}")
    print(f"      shape/dtype: {sanity['shape_ok']}/{sanity['dtype_ok']} "
          f"([3,{config.INPUT_SIZE},{config.INPUT_SIZE}], float32) | "
          f"normalization OK: {sanity['normalization_ok']}")
    print(f"      photo stats (informational): mean {sanity['photo_avg_channel_mean']}, "
          f"std {sanity['photo_avg_channel_std']}")

    # visualization + machine-readable report
    preview_path = config.RESULTS_PREP / "pipeline_preview.png"
    preview_grid(preview_path)
    report["preview_path"] = str(preview_path)

    report_path = config.RESULTS_PREP / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved -> {report_path}")
    print(f"Saved -> {preview_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
