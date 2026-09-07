"""Algorithm 7 - Lane detection.

    LaneDetector(mode='classical')  -> uses OpenCV (Canny + Hough + poly fit),
                                       works out of the box, no model needed.
    LaneDetector(mode='dl',        -> DeepLane / UFLD-style scaffold,
                weights=...)         trainable on TuSimple.

Both implement .predict(image) -> dict, with the same JSON contract.
"""

from .model import LaneDetector

__all__ = ["LaneDetector"]
