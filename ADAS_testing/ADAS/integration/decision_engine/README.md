# Algorithm 11 - Decision Engine / Result Aggregation

Pure-rule fusion that takes:

- AI results from the webcam pipeline (or any other source with the same dict shape)
- ESP32 sensor data (distance, rain, temperature, humidity, ambient light)

and produces structured **`ADAS`** warnings.

## Files
- `engine.py`    `DecisionEngine().evaluate(ai_results, sensor_data) -> dict`
- `config.py`    Warning catalogue + thresholds
- `demo.py`      Two fake scenarios (rainy day; foggy night + obstacle close)

## Quick check without any real ESP32
```
python integration/decision_engine/demo.py
```

## Integrates in the WebcamPipeline result like this
```python
from integration.decision_engine import DecisionEngine
engine = DecisionEngine()
result = engine.evaluate(webcampipeline.step_frame(frame), esp32.data())
# result["warnings"] -> list of "OBSTACLE_AHEAD", "LOW_VISIBILITY", ...
```

## Warning codes
| Code | When | Priority |
|---|---|---|
| OBSTACLE_AHEAD | ultrasonic < threshold | HIGH |
| RAIN_DETECTED  | model `rainy` or sensor rain=true or humidity > threshold | LOW |
| LOW_VISIBILITY | model `foggy/snowy/overcast` (conf > threshold) | HIGH |
| PEDESTRIAN_DETECTED | YOLO `person` with conf>threshold | HIGH |
| VEHICLE_PROXIMITY   | YOLO car/truck/bus/motorbike/conf>=thr | MEDIUM |
| LANE_DEPARTURE | num_lanes <= 1 (or 0) | HIGH |
| TRAFFIC_SIGN | sign class with conf>threshold | INFO |
| STOP_SIGN_DETECTED | as above + name contains 'stop' | HIGH |
| LOW_LIGHT / SUNGLARE | ambient_light < low / > high | MEDIUM |
| TEMPERATURE_HIGH | temperature_c > 40 | INFO |

## Limits
- Rule-based: does not classify how *critical* a warning is, only that it
  triggered. Production ADAS would use temporal filtering (warning only
  when the condition persists across N frames). This is fine for a
  college-proto / research project; not for safety-critical use.
