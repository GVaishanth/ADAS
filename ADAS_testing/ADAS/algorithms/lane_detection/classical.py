"""
classical.py - OpenCV-based lane detector (no training needed).

Pipeline: grayscale -> gaussian -> Canny -> region-of-interest mask ->
Hough lines -> slope-based left/right split -> least-squares polynomial fit
per group -> sample N points along each fit.
"""

from pathlib import Path

import cv2
import numpy as np

from . import config


def _canny(gray):
    blur = cv2.GaussianBlur(gray, (config.GAUSSIAN_KERNEL, config.GAUSSIAN_KERNEL), 0)
    return cv2.Canny(blur, config.CANNY_LOW, config.CANNY_HIGH)


def _roi_mask(shape):
    H, W = shape[:2]
    top_y    = int(H * config.DEFAULT_ROI_VB_RATIO[0])
    bottom_y = int(H * config.DEFAULT_ROI_VB_RATIO[1])
    pts = np.array([[
        (int(W * 0.10), bottom_y),
        (int(W * 0.45), top_y),
        (int(W * 0.55), top_y),
        (int(W * 0.90), bottom_y),
    ]], dtype=np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _hough_lines(edge, mask):
    masked = cv2.bitwise_and(edge, mask)
    theta  = config.HOUGH_THETA or np.pi / 180
    return cv2.HoughLinesP(
        masked,
        rho=config.HOUGH_RHO,
        theta=theta,
        threshold=config.HOUGH_THRESH,
        minLineLength=config.HOUGH_MIN_LINE,
        maxLineGap=config.HOUGH_MAX_GAP,
    )


def _split_lrl(lines):
    """Bucket lines into (left, right) by slope sign and filter ~vertical ones."""
    left, right = [], []
    if lines is None:
        return left, right
    # OpenCV HoughLinesP is (N,1,4) in 4.x but (N,4) in 5.x; accept both.
    lines = np.asarray(lines).reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        if abs(slope) < 0.3:                  # too horizontal -> ignore
            continue
        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))
    return left, right


def _fit_polys(lines, img_shape, n_points):
    """Fit a 1st-degree (and optionally 2nd-degree) polynomial per lane group,
    return list of dicts {slope_intercept, points=[[x,y]...]}.
    Returns at most 2 lanes (left, right)."""
    H, W = img_shape[:2]
    y_eval_top = int(H * config.DEFAULT_ROI_VB_RATIO[0])
    y_eval_bot = int(H * config.DEFAULT_ROI_VB_RATIO[1])
    ys = np.linspace(y_eval_top, y_eval_bot, n_points)

    polys = []
    for group in (lines[0], lines[1]):       # left then right
        if not group:
            continue
        xs_all = np.array([g[0] for g in group] + [g[2] for g in group])
        ys_all = np.array([g[1] for g in group] + [g[3] for g in group])
        # first-degree fit (good enough for a robust prototype)
        slope, intercept = np.polyfit(ys_all, xs_all, 1)
        x_pred = (slope * ys + intercept).astype(int).tolist()
        polys.append({
            "slope":     float(slope),
            "intercept": float(intercept),
            "points":    [[int(x), int(y)] for x, y in zip(x_pred, ys)],
        })
    return polys


def _clahe(gray):
    """CLAHE on the grayscale image: lifts contrast on dark / foggy / wet frames
    so lane markings survive adverse weather."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_lanes(bgr, n_points=None, enhance=True):
    """End-to-end classical lane detection -> list of lane point lists.

    enhance=True runs a CLAHE pre-pass so the detector also works on dark
    / foggy / rainy frames, not only bright daylight scenes.
    """
    n_points = n_points or config.NUM_PTS_PER_LANE
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if enhance:
        gray = _clahe(gray)
    edge = _canny(gray)
    mask = _roi_mask(edge.shape)
    lines = _hough_lines(edge, mask)
    left, right = _split_lrl(lines)
    lanes = _fit_polys((left, right), bgr.shape, n_points)
    return lanes, (left, right)


def draw_lanes(bgr, lanes, colors=None, thickness=4):
    """Return a copy of bgr with the lane polylines drawn."""
    out = bgr.copy()
    if not lanes:
        return out
    if colors is None:
        colors = [config.DEFAULT_LANE_COLOR_LEFT, config.DEFAULT_LANE_COLOR_RIGHT]
    for i, lane in enumerate(lanes):
        color = colors[i] if i < len(colors) else (255, 255, 255)
        for j in range(len(lane["points"]) - 1):
            pt1 = (lane["points"][j][0],     lane["points"][j][1])
            pt2 = lane["points"][j + 1]
            cv2.line(out, pt1, pt2, color, thickness)
    return out
