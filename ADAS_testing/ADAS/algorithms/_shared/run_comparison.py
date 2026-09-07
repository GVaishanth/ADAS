"""
run_comparison.py
=================
One-shot runner for the four-model comparison: fine-tune ViT / Swin /
ConvNeXt / EfficientFormer on the same BDD weather-ish split, then evaluate
each on the shared test split and assemble results/comparison.json.

Run from ADAS/ (cwd = ADAS/):
    python -m algorithms._shared.run_comparison --epochs 25

This makes the Task-1 "replace CNNs with ViT/Swin/ConvNeXt/EfficientFormer and
compare results" deliverable reproducible with a single command.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]     # ADAS/

MODELS = ["vit", "swin", "convnext", "efficientformer"]


def run(*args):
    """Run `python -m ...` with cwd=ADAS and pipe-through (quiet warnings)."""
    cmd = [str(sys.executable), "-m", *args]
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                       env={**__import__("os").environ,
                            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
                       capture_output=True, text=True)
    # print the interesting tail
    lines = (r.stdout + r.stderr).splitlines()
    keep = [l for l in lines if "Warning" not in l and "Loading" not in l
            and "it/s" not in l and "frames" not in l and "! Don't" not in l]
    for l in keep[-40:]:
        print(l)
    if r.returncode != 0:
        print("!! non-zero exit:", r.returncode)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pretrained", action="store_true",
                    help="ImageNet weights (FALSE by default = trains from scratch; "
                         "strongly recommended for the small bundled dataset, so this "
                         "comparison now defaults the recipe to pretrained via the "
                         "unified trainer).")
    ap.add_argument("--augment", action="store_true",
                    help="MixUp/CutMix + colour jitter + rotation (recommended on 80 imgs)")
    ap.add_argument("--step", default="all", choices=["all", "train", "eval"])
    args = ap.parse_args()

    model_dir = PROJECT_ROOT / "models"
    trained_ok, failed = [], []

    if args.step in ("all", "train"):
        # Unified, apples-to-apples trainer (single source of truth). Pretrained is
        # turned ON here because the old `--pretrained`-only path trained transformers
        # from scratch on 80 images --- the cause of the swin/convnext 0.5 collapse.
        for m in MODELS:
            rc = run(f"algorithms._shared.train_classifier",
                     "--backbone", m,
                     "--epochs", str(args.epochs),
                     "--pretrained",
                     "--augment" if args.augment else "")
            if rc == 0:
                trained_ok.append(m)
            else:
                failed.append(m)
                print(f"[run_comparison] training FAILED for backbone '{m}' (rc={rc}). "
                      "See the traceback above. It will be skipped, so its existing "
                      "comparison row (if any) is preserved and NOT overwritten with "
                      "random-weight numbers.")

    # Evaluate only backbones that actually HAVE a trained checkpoint. This
    # guarantees the comparison table is never filled with random-weight
    # ("PIPELINE VALIDATION ONLY") rows after a failed training run.
    to_eval = [m for m in MODELS if (model_dir / m / "best.pt").exists()]
    if args.step in ("all", "eval"):
        for m in to_eval:
            run(f"algorithms.{m}.evaluate")

    comp = PROJECT_ROOT / "results" / "comparison.json"
    print("\n=== FINAL comparison.json ===")
    print(comp.read_text() if comp.exists() else "NOT WRITTEN")
    if failed:
        print(f"\n[run_comparison] {len(failed)} backbone(s) failed to train and were "
              f"excluded from the comparison: {failed}. "
              "Fix the error and re-run; any existing trained rows for these were "
              "left untouched (not overwritten with random weights).")
    else:
        print("\n[run_comparison] all backbones trained and evaluated successfully.")


if __name__ == "__main__":
    main()
