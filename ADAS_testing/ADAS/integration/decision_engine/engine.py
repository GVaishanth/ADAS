"""
engine.py - rule-based DecisionEngine.

Combines:
  - AI results from the WebcamPipeline (weather, objects, lanes, signs, segments)
  - ESP32 sensor data (distance, rain, temperature, humidity, ambient light)

into structured warnings:

    {
       "status":  "safe" | "warning" | "danger",
       "warnings": [
          {"code": "RAIN_DETECTED",
           "priority": "LOW",
           "sources": ["weather:rainy", "sensor:rain=True"],
           "message": "rain detected from camera (94%) and ESP32 sensor"
           },
          ...
       ],
       "actions": ["ENABLE_RAIN_MODE", "WIPERS"],
       "raw":     { weather: ..., objects: ..., ... },      # compact of inputs
    }
"""

import time
import warnings
from typing import Dict, List

from . import config


# Re-exported: ADASWarning (alias) + thresholds reference
DEFAULT_THRESHOLDS = config.DEFAULT_THRESHOLDS
ADASWarning = config.ADASWarnings


class DecisionEngine:
    """Pure-rule aggregator. Anyone can swap reasoning later (e.g. learn decision
    boundaries) without changing the surrounding integration code."""

    def __init__(self, thresholds: Dict = None):
        self.thresholds = dict(thresholds or config.DEFAULT_THRESHOLDS)

    # ---------------------------------------------------------------- public
    def evaluate(self, ai_results: dict, sensor_data: dict) -> dict:
        """ai_results = output of WebcamPipeline.step(frame) (or a dict of the
        same shape). sensor_data = esp32 dict (see config.DEFAULT_*).

        Returns a JSON-friendly dict."""
        warnings = []
        actions  = set()
        sensor_data = sensor_data or {}

        # weather classification
        weather_class = self._safe_get(ai_results, "weather.prediction")
        weather_conf  = self._safe_get(ai_results, "weather.confidence")
        if weather_class and weather_conf is not None:
            conf_thr = self.thresholds["weather_model_conf_threshold"]
            if weather_conf >= conf_thr:
                warnings.extend(self._warn_weather(weather_class, sensor_data))

        # object detection
        for warn in self._warn_objects(ai_results, sensor_data):
            warnings.append(warn)

        # lane detection
        warnings.extend(self._warn_lanes(ai_results))

        # traffic-sign detection
        warnings.extend(self._warn_signs(ai_results))

        # sensor-only signals (rain / LDR / humidity / temperature)
        warnings.extend(self._warn_sensors(sensor_data))

        # collapse same warning codes
        merged = {}
        for w in warnings:
            code = w["code"]
            if code not in merged:
                merged[code] = dict(w); merged[code]["sources"] = set(w["sources"])
                merged[code]["messages"] = [w["message"]]
            else:
                merged[code]["sources"].update(w["sources"])
                merged[code]["messages"].append(w["message"])

        for w in merged.values():
            w["sources"]   = sorted(w["sources"])
            w["message"]   = "; ".join(w["messages"])
            w["priority"]  = config.ADASWarnings.get(w["code"], {}).get("priority", "INFO")
            actions.update(config.ADASWarnings.get(w["code"], {}).get("actions", []))

        warn_list = sorted(merged.values(),
                           key=lambda r: {"HIGH": 0, "MEDIUM": 1, "INFO": 2, "LOW": 3}
                                          .get(r.get("priority"), 4))
        status = "safe"
        priorities = [w["priority"] for w in warn_list]
        if "HIGH" in priorities:          status = "danger"
        elif "MEDIUM" in priorities:      status = "warning"

        return {
            "status":    status,
            "warnings":  warn_list,
            "actions":   sorted(actions),
            "timestamp": round(time.time(), 3),
            "raw": {
                "weather_class":  weather_class,
                "weather_conf":   weather_conf,
                "signs":          self._safe_get(ai_results, "signs.prediction"),
                "object_counts":  self._count_by_class(ai_results),
                "lane_count":     self._safe_get(ai_results, "lanes.results.num_lanes"),
                "sensors":        {k: sensor_data.get(k) for k in
                                     ("distance_cm","rain","temperature_c",
                                       "humidity_percent","ambient_light")},
            },
            "thresholds": self.thresholds,
        }

    # ----------------------------------------------------------- private helpers
    def _warn_weather(self, weather_class, sensor_data):
        ws = []
        if weather_class == "rainy":
            ws.append(self._mk("RAIN_DETECTED",
                                ["weather:rainy"],
                                "rain detected by camera (model prediction)"))
        if weather_class in ("foggy", "snowy", "overcast"):
            ws.append(self._mk("LOW_VISIBILITY",
                                [f"weather:{weather_class}"],
                                f"low visibility: weather classified as {weather_class}"))
        return ws

    def _warn_objects(self, ai, sensors):
        ws = []
        det    = ai.get("objects", {}).get("results", {}).get("detections", []) or []
        thr_min = self.thresholds["yolo_conf_threshold"]
        counts  = {}
        for d in det:
            c = d["class"]; counts[c] = counts.get(c, 0) + 1
            if d.get("confidence", 0.0) < thr_min:
                continue
            if c in {"person", "pedestrian"}:
                ws.append(self._mk("PEDESTRIAN_DETECTED",
                                    [f"object:{c}"],
                                    f"pedestrian detected (conf {d['confidence']:.2f})"))
            elif c in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}:
                ws.append(self._mk("VEHICLE_PROXIMITY",
                                    [f"object:{c}"],
                                    f"{c} detected ({d['confidence']:.2f})"))
        # If ESP32 ultrasonic detects something close, add OBSTACLE_AHEAD
        dist = sensors.get("distance_cm")
        if dist is not None and dist < self.thresholds["obstacle_distance_cm"]:
            ws.append(self._mk("OBSTACLE_AHEAD",
                                [f"sensor:distance_cm={dist}"],
                                f"obstacle {dist} cm ahead"))
        return ws

    def _warn_lanes(self, ai):
        ws = []
        n = ai.get("lanes", {}).get("results", {}).get("num_lanes", 0)
        if n <= self.thresholds["lane_count_too_few"]:
            ws.append(self._mk("LANE_DEPARTURE",
                                [f"lanes:num={n}"],
                                f"lane tracking lost (num_lanes={n})"))
        return ws

    def _warn_signs(self, ai):
        ws = []
        sign   = ai.get("signs", {}).get("prediction")
        sign_c = ai.get("signs", {}).get("confidence")
        thr = self.thresholds["sign_conf_threshold"]
        if sign and sign_c and sign_c >= thr:
            ws.append(self._mk("TRAFFIC_SIGN",
                                [f"signs:{sign}"],
                                f"traffic sign: {sign} ({sign_c*100:.1f}%)"))
            if "stop" in str(sign).lower():
                ws.append(self._mk("STOP_SIGN_DETECTED",
                                    [f"signs:{sign}"],
                                    "stop sign - prepare to stop"))
        return ws

    def _warn_sensors(self, sensors):
        ws = []
        rain_sensor_active = sensors.get("rain")
        if rain_sensor_active is self.thresholds["rain_sensor_active"]:
            ws.append(self._mk("RAIN_DETECTED",
                                [f"sensor:rain={rain_sensor_active}"],
                                "rain reported by ESP32 sensor"))
        light = sensors.get("ambient_light")
        if light is not None and light < self.thresholds["low_light_lux"]:
            ws.append(self._mk("LOW_LIGHT",
                                [f"sensor:ambient_light={light}"],
                                f"ambient light {light} lux (low)"))
        if light is not None and light > self.thresholds["sun_glare_lux"]:
            ws.append(self._mk("SUNGLARE",
                                [f"sensor:ambient_light={light}"],
                                f"ambient light {light} lux (very bright)"))
        hum = sensors.get("humidity_percent")
        if hum is not None and hum > self.thresholds["humidity_rain_threshold"]:
            ws.append(self._mk("RAIN_DETECTED",
                                [f"sensor:humidity={hum}%"],
                                f"humidity {hum}% suggests possible precipitation"))
        temp = sensors.get("temperature_c")
        if temp is not None and temp > 40:
            ws.append(self._mk("TEMPERATURE_HIGH",
                                [f"sensor:temperature_c={temp}"],
                                f"temperature {temp} C"))
        return ws

    @staticmethod
    def _mk(code, sources, message):
        return {"code": code, "sources": sources, "message": message}

    @staticmethod
    def _safe_get(d, dotted, default=None):
        cur = d
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @staticmethod
    def _count_by_class(ai):
        det = ai.get("objects", {}).get("results", {}).get("detections", []) or []
        out = {}
        for d in det:
            out[d["class"]] = out.get(d["class"], 0) + 1
        return out
