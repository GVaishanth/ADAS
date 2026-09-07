"""Decision engine configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_OUT  = RESULTS_DIR / "decision_engine"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

# Thresholds - safe default values; tune later. All distances are cm.
DEFAULT_THRESHOLDS = {
    "obstacle_distance_cm":         150,        # OBSTACLE_AHEAD when ultrasonic < this
    "low_light_lux":                 80,        # LOW_LIGHT when ambient light < this
    "sun_glare_lux":                800,        # SUNGLARE when ambient light > this
    "rain_sensor_active":          True,        # active when this flag is True
    "weather_model_conf_threshold": 0.55,       # only trust weather when above
    "weather_model_consensus":      False,      # if True, require >=2 weather models to agree
    "yolo_conf_threshold":           0.45,      # minimum YOLO confidence to trigger
    "sign_conf_threshold":           0.45,      # minimum traffic-sign confidence
    "lane_lost":                   False,       # True => trigger LANE_DEPARTURE
    "lane_count_too_few":            1,         # <= this many lanes triggers warning
    "humidity_rain_threshold":      85,         # humidity > this => possible rain
}


# Warning catalogue - keeps the JSON stable even if we add more warnings later
ADASWarnings = {
    "OBSTACLE_AHEAD":     {"priority": "HIGH",   "actions": ["REDUCE_SPEED", "BRAKE"]},
    "RAIN_DETECTED":      {"priority": "LOW",    "actions": ["ENABLE_RAIN_MODE", "WIPERS"]},
    "LOW_VISIBILITY":     {"priority": "HIGH",   "actions": ["ENABLE_FOG_LIGHTS", "REDUCE_SPEED"]},
    "PEDESTRIAN_DETECTED":    {"priority": "HIGH",   "actions": ["BRAKE"]},
    "VEHICLE_PROXIMITY":  {"priority": "MEDIUM", "actions": ["BRAKE"]},
    "LANE_DEPARTURE":     {"priority": "HIGH",   "actions": ["CORRECTION_REQ"]},
    "LANE_LOST":          {"priority": "MEDIUM", "actions": ["USE_HEADLIGHTS"]},
    "TRAFFIC_SIGN":       {"priority": "INFO",   "actions": ["NOTIFY_DRIVER"]},
    "STOP_SIGN_DETECTED": {"priority": "HIGH",   "actions": ["STOP_INSTRUCTION"]},
    "LOW_LIGHT":          {"priority": "MEDIUM", "actions": ["USE_HEADLIGHTS"]},
    "SUNGLARE":           {"priority": "MEDIUM", "actions": ["SHADE_VISOR"]},
    "TEMPERATURE_HIGH":   {"priority": "INFO",   "actions": ["AC_ON"]},
    "SENSOR_OFFLINE":     {"priority": "INFO",   "actions": ["RETRY_SENSOR"]},
    "MODEL_ERROR":        {"priority": "LOW",    "actions": ["LOG_ONLY"]},
}


def get_device():
    return "cpu"   # decision engine is rule-based - no GPU
