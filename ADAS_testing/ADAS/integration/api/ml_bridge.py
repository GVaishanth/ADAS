"""
ml_bridge.py
=============
ADAPTER between the frontend's `DetectionPacket` shape and our ML algorithms.

The frontend (cloned at frontend_repo, untouched) reads:

    window.dispatchEvent(new CustomEvent('adas-detection', { detail: detectionPacket }))

where `detectionPacket` matches the TypeScript `DetectionPacket` type:

    source: 'model' | 'demo'
    vehicle: { detected, distanceM, position: 'front'|'rear-left'|'rear-right' } | null
    rain: boolean
    lane: 'steady' | 'changing-left' | 'changing-right' | null
    trafficLight: 'red' | 'yellow' | 'green' | null
    speedLimit: number | null
    roadSurface: 'dry' | 'wet' | null
    gear: 'P' | 'R' | 'N' | 'D'
    speedKph: number
    objects: [{ label, x, y, width, height }]   (all percentages 0..100)

This module:
  - ML_TO_FRONTEND.convert(...)             converts a dict built from
    {WebcamPipeline.step(...) + esp32_data} into that exact JSON shape.
  - Mapping rules documented inline (e.g. weather "rainy" -> rain=True).

The result is what the FastAPI endpoint /detect returns to the browser.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence


# Display label only for road-relevant COCO classes.
COCO_ROAD_CLASSES = {
    0:  "Person",
    1:  "Cyclist",
    2:  "Vehicle",
    3:  "Motorbike",
    5:  "Bus",
    7:  "Truck",
    9:  "Signal",
    11: "Stop sign",
}

# Map our algorithm names to the labels the dashboard wants.
COCO_NAME_TO_LABEL = {
    "car":            "Vehicle",
    "truck":          "Truck",
    "bus":            "Bus",
    "motorcycle":     "Motorbike",
    "bicycle":        "Cyclist",
    "person":         "Person",
    "traffic light":  "Signal",
    "stop sign":      "Stop sign",
}

# GTSRB Class IDs that are speed-limit signs and their numeric values.
GTSRB_SPEED_LIMIT_IDS = {
    0:   20,    # Speed limit (20km/h)
    1:   30,
    2:   50,
    3:   60,
    4:   70,
    5:   80,
    7:   100,
    8:   120,
}


def _bbox_to_pct(bbox: Sequence[float], img_w: int, img_h: int) -> Dict[str, float]:
    """Convert (x1,y1,x2,y2) pixel bbox into percent terms the React UI uses."""
    x1, y1, x2, y2 = bbox
    x  = max(0.0, min(100.0, 100.0 * x1 / max(img_w, 1)))
    y  = max(0.0, min(100.0, 100.0 * y1 / max(img_h, 1)))
    w  = max(0.0, min(100.0, 100.0 * (x2 - x1) / max(img_w, 1)))
    h  = max(0.0, min(100.0, 100.0 * (y2 - y1) / max(img_h, 1)))
    return {"x": x, "y": y, "width": w, "height": h}


def _lane_state(num_lanes: Optional[int]) -> Optional[str]:
    if num_lanes is None:
        return None
    if num_lanes <= 1:
        return "changing-left"        # 'lane tracking lost' => UI shows 'changing'
    return "steady"


def _road_surface(weather_class: Optional[str],
                  rain_sensor:    Optional[bool]) -> Optional[str]:
    if rain_sensor is True:                  return "wet"
    if weather_class in {"rainy", "snowy"}:  return "wet"
    if weather_class in {"foggy", "overcast",
                          "clear", "partly cloudy"}:
        return "dry"
    return None


def _has_tlight(dets: List[Dict[str, Any]]) -> bool:
    return any(d.get("class") == "traffic light" for d in dets)


def _traffic_light_color(bgr, dets: List[Dict[str, Any]]) -> Optional[str]:
    """Detect whether a detected 'traffic light' is red / yellow / green.

    YOLO gives us the box(es) but not the active lamp. We sample the pixel
    colours inside the box and pick the dominant red / yellow / green hue.
    Returns None (unknown) when there is no clear dominant lamp colour.
    """
    if bgr is None:
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    boxes = [d.get("bbox") for d in dets
             if d.get("class") == "traffic light" and d.get("bbox")]
    if not boxes:
        return None
    h, w = bgr.shape[:2]
    votes = {"red": 0.0, "yellow": 0.0, "green": 0.0}
    for b in boxes:
        try:
            x1, y1, x2, y2 = (int(v) for v in b)
            x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2)); y2 = max(y1 + 1, min(h, y2))
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            # only saturated, reasonably bright pixels (the lit lamp)
            bright = (sat > 90) & (val > 90)
            # OpenCV hue is 0..179; red wraps around 0.
            # np.count_nonzero() returns a plain Python int and is robust across
            # numpy 2.x (some boolean-reduction paths raise on float()/sum()).
            votes["red"] += float(np.count_nonzero(((hue <= 10) | (hue >= 165)) & bright))
            votes["yellow"] += float(np.count_nonzero((hue >= 20) & (hue <= 35) & bright))
            votes["green"] += float(np.count_nonzero((hue >= 40) & (hue <= 90) & bright))
        except Exception:
            continue

    total = sum(votes.values())
    if total < 8:                          # too little signal -> unknown
        return None
    best = max(votes, key=votes.get)
    if votes[best] / total < 0.45:         # ambiguous -> don't commit to a colour
        return None
    return best


def from_ai_and_sensors(
    ai_results:  Dict[str, Any],
    sensor_data: Dict[str, Any] | None,
    img_w: int,
    img_h: int,
    image=None,
) -> Dict[str, Any]:
    """Translate one AI step + sensor data into the frontend's DetectionPacket shape."""
    sensor_data = sensor_data or {}
    distance_cm = sensor_data.get("distance_cm")
    rain_sensor = sensor_data.get("rain")
    gear        = sensor_data.get("gear") or "D"
    speed_kph   = sensor_data.get("speed_kph") or 0

    weather_class = (ai_results.get("weather", {}) or {}).get("prediction")
    weather_conf  = (ai_results.get("weather", {}) or {}).get("confidence")
    objects_ai    = ((ai_results.get("objects", {}) or {}).get("results", {}) or {}).get("detections", []) or []
    lane_count    = (ai_results.get("lanes",   {}) or {}).get("results", {}).get("num_lanes")
    signs_pred    = (ai_results.get("signs", {}) or {}).get("prediction")
    signs_conf    = (ai_results.get("signs", {}) or {}).get("confidence")
    signs_classid = (ai_results.get("signs", {}) or {}).get("results", {}).get("class_id")

    # ---- vehicle block ------------------------------------------------
    front_obj = next((d for d in objects_ai
                       if d.get("class") in {"car","truck","bus","motorcycle"}), None)
    try:
        distance_m = round(float(distance_cm) / 100.0, 2) if distance_cm is not None else 0.0
    except Exception:
        distance_m = 0.0
    vehicle = {
        "detected":  front_obj is not None,
        "distanceM": distance_m,
        "position":  "front",
    }

    # ---- weather/rain -------------------------------------------------
    rain = bool(rain_sensor) or (weather_class == "rain")

    # ---- lane ---------------------------------------------------------
    lane = _lane_state(lane_count)

    # ---- traffic light ------------------------------------------------
    # Prefer the colour sampled from the image; fall back to the old
    # behaviour (green) only when we couldn't read a lamp colour.
    traffic_light: Optional[str] = _traffic_light_color(image, objects_ai)
    if traffic_light is None and _has_tlight(objects_ai):
        traffic_light = "green"

    # ---- speed limit from GTSRB ---------------------------------------
    speed_limit: Optional[int] = None
    if isinstance(signs_classid, int) and signs_classid in GTSRB_SPEED_LIMIT_IDS:
        speed_limit = GTSRB_SPEED_LIMIT_IDS[signs_classid]

    # ---- road surface -------------------------------------------------
    road_surface = _road_surface(weather_class, rain_sensor)

    # ---- detection frames --------------------------------------------
    obj_list: List[Dict[str, Any]] = []
    for d in objects_ai[:8]:
        label = COCO_NAME_TO_LABEL.get(d.get("class", ""), d.get("class", "").title() or "Object")
        pct = _bbox_to_pct(d.get("bbox", [0, 0, 0, 0]), img_w, img_h)
        obj_list.append({"label": label, **pct})

    return {
        "source":        "model",
        "vehicle":       vehicle,
        "rain":          rain,
        "lane":          lane,
        "trafficLight":  traffic_light,
        "speedLimit":    speed_limit,
        "roadSurface":   road_surface,
        "gear":          gear,
        "speedKph":      speed_kph,
        "objects":       obj_list,
        # Internal fields (frontend ignores them):
        "_ts":           round(time.time(), 3),
        "_weather":      weather_class,
        "_weather_conf": weather_conf,
        "_signs":        signs_pred,
    }
