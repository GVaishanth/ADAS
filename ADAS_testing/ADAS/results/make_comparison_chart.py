"""
make_comparison_chart.py
========================
Reads results/comparison.json and writes a grouped bar chart comparing the four
backbones on Accuracy / Precision / Recall / F1 / latency / size.

Run from ADAS/:
    python results/make_comparison_chart.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
COMP = PROJ / "results" / "comparison.json"


def main():
    if not COMP.exists():
        print("comparison.json not found — run the backbones + evaluate first.")
        return
    rows = json.loads(COMP.read_text())
    if not rows:
        print("comparison.json is empty.")
        return
    rows = sorted(rows, key=lambda r: -float(r.get("accuracy", 0)))
    labels = [r["model"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # accuracy / precision / recall / f1
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    width = 0.2
    colors = ["#2563eb", "#06b6d4", "#34d399", "#a78bfa"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [float(r.get(m, 0)) * 100 for r in rows]
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=m, color=c)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("%"); axes[0].set_title("Per-backbone classification metrics")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.3)

    # latency
    lat = [float(r.get("inference_time_ms", 0)) for r in rows]
    axes[1].bar(labels, lat, color="#f59e0b")
    axes[1].set_ylabel("ms / image"); axes[1].set_title("Inference latency (CPU, 1 thread)")
    axes[1].grid(axis="y", alpha=.3)
    for i, v in enumerate(lat):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # fps + size
    fps = [float(r.get("fps", 0)) for r in rows]
    size = [float(r.get("model_size_mb", 0)) for r in rows]
    axes[2].bar(labels, fps, color="#6366f1", label="FPS")
    axes[2].set_ylabel("FPS"); axes[2].set_title("Throughput (FPS) / model size")
    axes[2].legend(); axes[2].grid(axis="y", alpha=.3)
    for i, v in enumerate(fps):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    out = PROJ / "results" / "comparison_chart.png"
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")

    # also a human-readable summary
    print("\nModel                Acc     Prec    Rec     F1      lat(ms) FPS   size(MB)  ckpt")
    print("-" * 95)
    for r in rows:
        print(f"{r['model']:20s} {float(r.get('accuracy',0))*100:6.2f}% "
              f"{float(r.get('precision',0))*100:6.2f}% {float(r.get('recall',0))*100:6.2f}% "
              f"{float(r.get('f1',0))*100:6.2f}% {float(r.get('inference_time_ms',0)):7.2f} "
              f"{float(r.get('fps',0)):5.1f} {float(r.get('model_size_mb',0)):8.1f}  {r.get('checkpoint','')}")


if __name__ == "__main__":
    main()
