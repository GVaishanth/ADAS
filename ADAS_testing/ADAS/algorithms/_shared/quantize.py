"""
quantize.py
===========
Export a trained timm backbone to ONNX and optionally int8-quantize it, then
measure per-image latency and model size. This is the "deploy faster" lever
for the classifier (Swin/ConvNeXt currently the slowest at ~80–116 ms).

Run from ADAS/ (cwd = ADAS/):
    python -m algorithms._shared.quantize --backbone vit --int8
    python -m algorithms._shared.quantize --backbone swin --int8

Produces:
    models/<name>/model.onnx               (fp32 export)
    models/<name>/model_int8.onnx          (if --int8 and onnxruntime available)
    models/<name>/quantize_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

BACKBONES = {
    "vit": "algorithms.vit",
    "swin": "algorithms.swin",
    "convnext": "algorithms.convnext",
    "efficientformer": "algorithms.efficientformer",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), default="vit")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--input-size", type=int, default=224)
    args = ap.parse_args()

    mod_name = BACKBONES[args.backbone]
    cfg = __import__(f"{mod_name}.config", fromlist=["config"])
    model_cls = getattr(__import__(f"{mod_name}.model", fromlist=["Classifier"]),
                        f"{args.backbone.capitalize()}Classifier")
    device = cfg.get_device()

    model = model_cls(num_classes=cfg.NUM_CLASSES, pretrained=False)
    ckpt = cfg.MODEL_OUT / "best.pt"
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}; training from scratch won't reflect real "
              f"accuracy, but export/latency still works. Train first for a valid export.")
    else:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)
    fp32_path = cfg.MODEL_OUT / "model.onnx"

    def export(path):
        try:
            torch.onnx.export(model, dummy, str(path), input_names=["input"],
                              output_names=["logits"], dynamic_axes={"input": {0: "b"}},
                              opset_version=17)
            return True
        except Exception as e:
            print("  ONNX export failed:", e)
            return False

    def latency(fn, runs):
        # warmup
        for _ in range(3):
            fn()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))

    report = {"backbone": args.backbone, "fp32_path": str(fp32_path),
              "int8": args.int8}

    if export(fp32_path):
        try:
            import onnxruntime as ort
        except ImportError:
            ort = None
        if ort is not None:
            def run_fp32():
                sess = ort.InferenceSession(str(fp32_path),
                                            providers=["CPUExecutionProvider"])
                return sess
            sess = run_fp32()
            report["onnx_fp32_ms"] = latency(lambda: sess.run(None, {"input": dummy.numpy()}), args.runs)
        else:
            print("onnxruntime not installed; install with `pip install onnxruntime` "
                  "to measure export latency and enable int8.")

        # pytorch baseline
        with torch.no_grad():
            report["torch_ms"] = latency(lambda: model(dummy), args.runs)

        if args.int8:
            try:
                import onnxruntime.quantization as quant
                onnx_model = Path(fp32_path)
                int8_path = cfg.MODEL_OUT / "model_int8.onnx"
                try:
                    quant.quantize_dynamic(str(onnx_model), str(int8_path))
                except Exception as e:
                    print("  dynamic int8 failed, trying static default:", e)
                    quant.quantize_dynamic(str(onnx_model), str(int8_path))
                if int8_path.exists():
                    from onnxruntime import InferenceSession
                    sess8 = InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
                    report["onnx_int8_ms"] = latency(
                        lambda: sess8.run(None, {"input": dummy.numpy()}), args.runs)
                report["int8_path"] = str(int8_path)
            except ImportError:
                print("onnxruntime.quantization not available; skipping int8.")

    report = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()}
    cfg.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    with open(cfg.MODEL_OUT / "quantize_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
