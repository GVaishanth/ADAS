"""
demo.py - run the DecisionEngine on a few example scenarios (no real sensors needed).

This is the easiest way to verify the rule logic works before any ESP32 is connected.
"""

import json, sys, random
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.decision_engine.engine import DecisionEngine


def make_fake_scenario():
    return {
        "ai": {
            "weather": {"prediction": random.choice(["rainy", "snowy", "clear", "foggy"]),
                          "confidence": 0.85},
            "objects": {"results": {"detections": [
                {"class": "person",     "confidence": 0.81, "bbox": [10, 10, 40, 100]},
                {"class": "car",        "confidence": 0.71, "bbox": [120, 80, 220, 180]},
            ]}},
            "lanes":   {"results": {"num_lanes": 2}},
            "signs":   {"prediction": "Stop", "confidence": 0.92},
        },
        "sensor": {
            "distance_cm": 240, "rain": False,
            "temperature_c": 35, "humidity_percent": 30,
            "ambient_light": 700,
        },
    }


def make_low_vis_obstacle_scenario():
    return {
        "ai": {
            "weather": {"prediction": "foggy", "confidence": 0.74},
            "objects": {"results": {"detections": []}},
            "lanes":   {"results": {"num_lanes": 0}},
            "signs":   {"prediction": None, "confidence": None},
        },
        "sensor": {
            "distance_cm": 60, "rain": False,
            "temperature_c": 22, "humidity_percent": 95,
            "ambient_light": 60,
        },
    }


def main():
    print("=" * 60)
    print("Scenario 1: rainy day + a person ahead, safe lane count")
    print("=" * 60)
    print(json.dumps(DecisionEngine().evaluate(
        make_fake_scenario()["ai"], make_fake_scenario()["sensor"]), indent=2))

    print("=" * 60)
    print("Scenario 2: foggy night + obstacle close + lost lanes")
    print("=" * 60)
    print(json.dumps(DecisionEngine().evaluate(
        make_low_vis_obstacle_scenario()["ai"], make_low_vis_obstacle_scenario()["sensor"]),
        indent=2))


if __name__ == "__main__":
    main()
