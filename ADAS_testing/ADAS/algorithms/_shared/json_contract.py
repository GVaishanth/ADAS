"""
json_contract.py
================
Single JSON envelope used by every algorithm module. FastAPI just dumps it.

    {
      "status":   "success" | "error",
      "prediction": str | None,
      "confidence": float | None,
      "inference_time_ms": float,
      "results":  {...algorithm-specific...}
    }
"""

import time
from contextlib import contextmanager


def make_envelope(prediction=None, confidence=None, inference_time_ms=0.0,
                  status="success", results=None):
    """Build the standard JSON dict."""
    return {
        "status": status,
        "prediction": prediction,
        "confidence": confidence,
        "inference_time_ms": round(float(inference_time_ms), 3),
        "results": results or {},
    }


@contextmanager
def timed():
    """Block: with timed() as t:  ... result = make_envelope(inference_time_ms=t.ms)"""
    t0 = time.perf_counter()
    box = {"ms": 0.0}
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - t0) * 1000.0


def error_envelope(message, inference_time_ms=0.0):
    return make_envelope(prediction=None, confidence=None,
                         inference_time_ms=inference_time_ms,
                         status="error", results={"error": message})
