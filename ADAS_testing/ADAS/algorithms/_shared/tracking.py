"""
tracking.py
===========
Lightweight single-class IoU tracker + time-to-collision (TTC) estimator.

Why: raw per-frame detection boxes flicker and the live "vehicle collision"
alert can be noisy. Tracking lets the system only raise a collision alert for a
genuinely close, in-lane object that persists across frames, giving real
precision / recall gains in the UI without retraining anything.

API is deliberately dependency-free (numpy only) so it can run inside the
server's /detect path next to the models.

Example
-------
    tracker = IoUTracker(iou_thr=0.35, max_age=4)
    for frame in video:
        boxes = detector.predict(frame)  # [ {x,y,w,h,label,conf} ... ]
        tracks = tracker.update(boxes)
        for t in tracks: print(t["tracks_id"], t["ttc_s"])
"""

from __future__ import annotations

import math

import numpy as np


def compute_iou(a, b):
    """IoU of two [x, y, w, h] boxes (top-left origin, pixel or % coords)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-9
    return inter / union


class Track:
    """One persistent object with a smoothed box + history."""
    __slots__ = ("track_id", "label", "cls", "history", "age", "misses", "ttc_ema")

    def __init__(self, tid, label, box, conf, alpha=0.5):
        self.track_id = tid
        self.label = label
        self.cls = conf          # best detection confidence
        self.history = [(box, conf)]
        self.age = 0
        self.misses = 0
        self.ttc_ema = None

    @property
    def box(self):
        return self.history[-1][0]

    def smoothed_box(self, alpha=0.5):
        """Exponential moving average of recent boxes (removes jitter)."""
        if len(self.history) == 1:
            return self.box
        b = np.asarray(self.history[-1][0], dtype=np.float64)
        for past in reversed(self.history[:-1][-5:]):
            b = alpha * b + (1 - alpha) * np.asarray(past[0], dtype=np.float64)
        return b.tolist()

    def estimate_ttc(self, camera_fov_deg, frame_height_px, camera_height_m, vehicle_width_m=1.8):
        """Time-to-collision (seconds) from apparent box growth across frames.

        A simple perspective approximation: if an object of fixed physical size
        doubles in apparent height between two frames, distance to it halves.
        """
        if len(self.history) < 2 or self.history[-1][0][3] <= 0:
            return None
        h0 = self.history[0][0][3]
        h1 = self.history[-1][0][3]
        if h1 <= h0:            # object shrank -> moving away / not approaching
            return None
        # if the object appears to grow, how much time has passed between
        # first and last observation (assumed ~0.7s model cadence)
        dt = max(1e-3, (len(self.history) - 1)) * 0.7
        # distance shrinks roughly inversely with apparent size
        ratio = h0 / h1         # < 1
        ttc = dt * ratio / max(1 - ratio, 1e-6)
        return ttc


class IoUTracker:
    """Greedy IoU tracker with a max miss-age; assigns stable track ids."""

    def __init__(self, iou_thr=0.35, max_age=4, max_speed=None, frame_dt=0.7):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}
        self.frame_dt = frame_dt

    def _match(self, boxes):
        """Greedily match detections -> tracks (largest IoU first)."""
        used_track, used_det = set(), {}
        pairs = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(boxes):
                iou = compute_iou(tr.box, det["box"])
                if iou >= self.iou_thr:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)
        for iou, tid, di in pairs:
            if tid in used_track or di in used_det:
                continue
            used_track.add(tid)
            used_det[di] = tid
        return used_det

    def update(self, boxes):
        """boxes: list of {box:[x,y,w,h], label, conf}. Returns list of track dicts."""
        boxes = [dict({k: v for k, v in b.items()}) for b in boxes]
        for b in boxes:
            b["box"] = [float(x) for x in b["box"]]
        matched = self._match(boxes)
        active_ids = set()
        # update matched
        for di, tid in matched.items():
            det = boxes[di]
            tr = self.tracks[tid]
            tr.history.append((det["box"], det["conf"]))
            if len(tr.history) > 8:
                tr.history.pop(0)
            tr.cls = max(tr.cls, det["conf"])
            tr.age += 1
            tr.misses = 0
            active_ids.add(tid)
        # age out unmatched tracks, spawn new ones for unmatched detections
        for tid, tr in list(self.tracks.items()):
            if tid not in active_ids:
                tr.misses += 1
                if tr.misses > self.max_age:
                    del self.tracks[tid]
        for di, det in enumerate(boxes):
            if di not in matched:
                tr = Track(self.next_id, det["label"], det["box"], det["conf"])
                self.next_id += 1
                self.tracks[tr.track_id] = tr
                active_ids.add(tr.track_id)

        out = []
        for tid in sorted(active_ids):
            tr = self.tracks[tid]
            b = tr.smoothed_box()
            out.append({
                "track_id": tr.track_id,
                "label": tr.label,
                "confidence": round(float(tr.cls), 3),
                "box": [round(x, 2) for x in b],
                "ttc_s": tr.estimate_ttc(camera_fov_deg=70, frame_height_px=None, camera_height_m=None),
                "stable": tr.age >= 2,
            })
        return out
