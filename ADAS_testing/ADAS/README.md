# ADAS — AI-Based Driver Assistance System

**Project:** Road-scene understanding under adverse weather conditions using
modern Vision Transformer architectures + ADAS perception modules, on the
laptop with ESP32 sensors.

This repo implements **Algorithms 1 to 11** as a unified system. Each
algorithm is self-contained (you can run any one from the command line) but
they all share the same data layout, the same `predict(...)->dict` JSON
contract, and the same master `results/comparison.json`.

See `ROADMAP.md` for the original plan; the layout below is the final
project structure.

---

## 1. Final folder structure

```
ADAS/
├── README.md                                   (this file)
├── ROADMAP.md
├── requirements.txt
│
├── algorithms/
│   ├── _shared/                                (helpers used by many modules)
│   │   ├── __init__.py
│   │   ├── json_contract.py
│   │   ├── metrics.py
│   │   ├── training.py
│   │   ├── transform_helper.py
│   │   └── transformer_base.py
│   │
│   ├── preprocessing/          (Algorithm 1)
│   │   ├── __init__.py / config.py / preprocessing.py
│   │   ├── build_dataset.py      <- build the BDD weather subset
│   │   ├── evaluate.py           <- dataset + pipeline integrity
│   │   └── predict.py            <- image / video / webcam
│   │
│   ├── vit/                    (Algorithm 2)
│   ├── swin/                   (Algorithm 3)
│   ├── convnext/               (Algorithm 4)
│   ├── efficientformer/        (Algorithm 5)
│   │   └── … each has __init__.py / config.py / model.py / train.py
│   │       evaluate.py / predict.py / README.md
│   │
│   ├── yolo/                   (Algorithm 6)
│   │   └── … + bdd_to_yolo.py (BDD → YOLO-format converter)
│   │
│   ├── lane_detection/         (Algorithm 7)
│   │   └── … + classical.py + tusimple.py
│   │
│   ├── traffic_sign/           (Algorithm 8)
│   │   └── … + dataset.py
│   │
│   └── segmentation/           (Algorithm 9)
│       └── … SegFormer (HuggingFace)
│
├── integration/
│   ├── webcam/                 (Algorithm 10)
│   │   ├── __init__.py / config.py
│   │   ├── pipeline.py            (WebcamPipeline class)
│   │   └── main.py                (CLI: `python integration/webcam/main.py`)
│   │
│   └── decision_engine/        (Algorithm 11)
│       ├── __init__.py / config.py / engine.py
│       └── demo.py                (two fake scenarios)
│
├── datasets/
│   ├── bdd100k/README.md        (where YOU put the official download)
│   ├── make_synthetic_bdd.py    (fake 7-class BDD for w/o-download tests)
│   └── bdd100k_weather/         (manifest CSVs produced by Algorithm 1)
│
├── models/
│   └── {vit, swin, convnext,
│        efficientformer, yolo,
│        lane_detection, traffic_sign,
│        segmentation}/best.pt   (populated when YOU train)
│
└── results/
    ├── preprocessing/        sanity + preview PNG
    ├── vit, swin, convnext,
        efficientformer,
        yolo, lane_detection,
        traffic_sign,
        segmentation/         per-module reports + confusion matrices
    ├── comparison.json       shared metric table for the 4 comparative models
    └── integration/          webcam run videos
```

---

## 2. Master requirements.txt (current)

```
pip install -r requirements.txt
# Adds: numpy, pandas, opencv-python, Pillow, matplotlib, scikit-learn,
#       tqdm, PyYAML, torch, torchvision, timm, ultralytics, transformers,
#       sentencepiece, scipy
```

The next module (FastAPI) will only add `fastapi`, `uvicorn`, `pydantic`,
`python-multipart`. Nothing re-listed here.

---

## 3. Datasets required (per module)

| # | Module | Dataset | Source | Storage | Required? |
|---|---|---|---|---|---|
| 1 | preprocessing | **BDD100K** (recommended) or none | https://bdd-data.berkeley.edu | ~7.1 GB | optional — `datasets/make_synthetic_bdd.py` for tests |
| 2-5 | vit/swin/convnext/efficientformer | same BDD subset as #1 | — | uses 0 MB of new storage (manifests reference existing) | yes (or synthetic) |
| 6 | yolo detection | **COCO pretrained** (out-of-box) — OR BDD100K detection subset | https://cocodataset.org / BDD100K | ~6 MB pretrained + 0 if no fine-tune | optional |
| 7 | lane_detection | **none** (classical works out of box). For DL: TuSimple | https://github.com/TuSimple/tusimple-benchmark | ~5 GB | optional |
| 8 | traffic_sign | **GTSRB** | https://benchmark.ini.rub.de/gtsrb_dataset.html | ~300 MB | recommended |
| 9 | segmentation | **Cityscapes** pretrained model (auto-downloaded 15 MB). For fine-tune: Cityscapes or ACDC | https://www.cityscapes-dataset.com ; ACDC: https://acdc.vision.ee.ethz.ch | ~20 GB (Cityscapes) / ~1 GB (ACDC) | recommended |
| 10 | webcam | your laptop webcam | — | — | always |
| 11 | decision_engine | none (rule-based) | — | — | always |

### Quick-install plan if you only want to test the demo (skip training)

```
pip install -r requirements.txt
python datasets/make_synthetic_bdd.py   # generates a tiny fake BDD
```
The YOLOv8 weights (~6 MB) auto-download on first run.
The SegFormer weights (~15 MB) auto-download on first run.

---

## 4. Per-module commands

(Cwd = `ADAS/`. Arrows = outputs.)

### Algorithm 1 — Data Processing
```bash
python datasets/make_synthetic_bdd.py
python algorithms/preprocessing/build_dataset.py --train-per-class 400 --val-per-class 100 --test-per-class 100
python algorithms/preprocessing/evaluate.py
python algorithms/preprocessing/predict.py --image <path>.jpg
   → datasets/bdd100k_weather/{manifest_train,manifest_val,manifest_test}.csv
   → results/preprocessing/{evaluation_report.json, pipeline_preview.png, ...}
```

### Algorithms 2-5 — Comparative models (same UI, swap module name)
```bash
# Train
python algorithms/<m>/train.py --pretrained --epochs 5
   → models/<m>/best.pt

# Evaluate (writes results/<m>/metrics.json + confusion matrix, appends results/comparison.json)
python algorithms/<m>/evaluate.py
   → results/comparison.json (a row per model)

# Predict
python algorithms/<m>/predict.py --image <path>.jpg     # ONE annotated image
python algorithms/<m>/predict.py --source 0             # WEBCAM

# All of: vit, swin, convnext, efficientformer
```

### Algorithm 6 — YOLOv8
```bash
# Out-of-box (pretrained COCO weights; download on first run)
python algorithms/yolo/predict.py --image <path>.jpg
python algorithms/yolo/predict.py --source 0

# Fine-tune on BDD detection classes (optional)
python algorithms/yolo/bdd_to_yolo.py                # converts BDD to YOLO format
python algorithms/yolo/train.py \
        --data datasets/bdd100k_yolo/bdd100k.yaml \
        --epochs 30
python algorithms/yolo/evaluate.py \
        --weights models/yolo/<run>/weights/best.pt \
        --data    datasets/bdd100k_yolo/bdd100k.yaml
```

### Algorithm 7 — Lane Detection
```bash
# Classical (no training)
python algorithms/lane_detection/predict.py --image <path>.jpg
python algorithms/lane_detection/predict.py --source 0

# DL mode (requires TuSimple + train)
python algorithms/lane_detection/train.py \
        --tusimple-root datasets/tusimple  --epochs 10
python algorithms/lane_detection/predict.py --image <path>.jpg \
        --mode dl \
        --weights models/lane_detection/lane_unet.pt
```

### Algorithm 8 — Traffic Sign Recognition
```bash
python algorithms/traffic_sign/train.py --auto-download --epochs 8
python algorithms/traffic_sign/evaluate.py --weights models/traffic_sign/best.pt
python algorithms/traffic_sign/predict.py --image <path>.jpg
python algorithms/traffic_sign/predict.py --source 0
```

### Algorithm 9 — Semantic Segmentation
```bash
python algorithms/segmentation/predict.py --image <path>.jpg     # auto-downloads SegFormer-B0 ~15 MB
python algorithms/segmentation/predict.py --source 0
python algorithms/segmentation/evaluate.py \
        --cityscapes-root datasets/cityscapes --max-images 50
```

### Algorithm 10 — Webcam Integration Pipeline
```bash
python integration/webcam/main.py --source 0             # all default models
python integration/webcam/main.py --source 0 \
        --no-yolo --no-seg                                # lighter run
python integration/webcam/main.py --source path/to/video.mp4 --out results/integration/my_run.mp4
```

### Algorithm 11 — Decision Engine (rule-based)
```bash
python integration/decision_engine/demo.py               # two fake scenarios, no webcam needed
```

---

## 5. Master checklist — what to run, in what order

Mark each box as you go. All 11 modules will use this order.

### A. Pre-flight (5 minutes)
- [ ] `cd ADAS`
- [ ] `pip install -r requirements.txt`
- [ ] Confirm you can `import torch, torchvision, timm, ultralytics, transformers, cv2`

### B. Algorithm 1 — Preprocessing
- [ ] (optional) `python datasets/make_synthetic_bdd.py` for the no-download path
- [ ] PUT your real BDD100K + JSON labels in `datasets/bdd100k/` — OR use synthetic
- [ ] `python algorithms/preprocessing/build_dataset.py --train-per-class <N>` (try smaller N first)
- [ ] `python algorithms/preprocessing/evaluate.py` → expect `normalization OK: True`, splits disjoint, balance OK
- [ ] `python algorithms/preprocessing/predict.py --image <path>.jpg`

### C. Algorithms 2-5 — train the four comparative models (each)
- [ ] `python algorithms/<m>/train.py --pretrained --epochs 5` for m in {vit, swin, convnext, efficientformer}
- [ ] `python algorithms/<m>/evaluate.py` for m in {vit, swin, convnext, efficientformer}
- [ ] Verify `results/comparison.json` now has 4 rows — the columns match the spec table
- [ ] Verify all four checkpoints are under `models/<m>/best.pt`

### D. Algorithm 6 — YOLOv8
- [ ] Out-of-box: `python algorithms/yolo/predict.py --image <path>.jpg`. Should detect at least 1 object in a road image.
- [ ] Optional fine-tune: `bdd_to_yolo.py` → `train.py` → `evaluate.py`

### E. Algorithm 7 — Lane Detection
- [ ] Classical: `python algorithms/lane_detection/predict.py --image <path>.jpg` → overlay saved
- [ ] Optional DL mode: `algorithms/lane_detection/train.py` then `--mode dl`

### F. Algorithm 8 — Traffic Sign Recognition
- [ ] (optional) `python algorithms/traffic_sign/train.py --auto-download` (~3 GB cache)
- [ ] `python algorithms/traffic_sign/evaluate.py` once checkpoint is ready

### G. Algorithm 9 — Semantic Segmentation
- [ ] `python algorithms/segmentation/predict.py --image <path>.jpg` (auto-downloads SegFormer ~15 MB; overlay & mask saved)

### H. Algorithm 10 — Webcam Integration
- [ ] `python integration/webcam/main.py --source 0` (ESC to stop; MP4 file is written)
- [ ] Try with `--no-seg` for a faster pipeline

### I. Algorithm 11 — Decision Engine
- [ ] `python integration/decision_engine/demo.py` — confirm 2 warnings on Scenario 2

### J. Joint verification (the research deliverable)
- [ ] Open `results/comparison.json` and replace **TBD** cells with your real numbers
- [ ] Make a chart from it (bar chart of Accuracy / F1 / Inference speed across the 4 models)
- [ ] Verify the warning count produced by `DecisionEngine().evaluate(step_result)` aligns with what the dashboard should show

---

## 6. Common FastAPI-ready interface (every algorithm class)

All four transformer predictors expose the same shape:
```python
from algorithms.vit           import ViTPredictor
pred = ViTPredictor(weights_path="models/vit/best.pt")
result = pred.predict(image_or_ndarray)
#  -> {
#       "status": "success", "prediction": "rainy",
#       "confidence": 0.94, "inference_time_ms": 23.4,
#       "results": {"top_k": [...], "model": ..., "logits_dim": 6}
#     }

from algorithms.yolo           import YOLODetector
from algorithms.lane_detection import LaneDetector
from algorithms.traffic_sign   import TrafficSignPredictor
from algorithms.segmentation   import SegmentationPredictor
from integration.webcam.pipeline             import WebcamPipeline
from integration.decision_engine.engine       import DecisionEngine

# In FastAPI:
#   @app.post("/predict/weather")     -> weather_pred.predict(image)
#   @app.post("/predict/objects")     -> obj_det.predict(image)
#   @app.post("/predict/lanes")       -> lane_det.predict(image)
#   @app.post("/predict/signs")       -> sign_pred.predict(image)
#   @app.post("/predict/segmentation")-> seg_pred.predict(image)
#   @app.post("/step")                -> pipeline.step(image)
#   /health                            : status
```
The **prediction/confidence/inference_time_ms/results** envelope is identical
across every algorithm, so the FastAPI handler is one copy-paste with two
different class instantiations.

---

## 7. ESP32 ↔ FastAPI data shape

Sent by the ESP32 over Wi-Fi (HTTP POST or MQTT):
```json
{
  "distance_cm":      240,
  "rain":             false,
  "temperature_c":    22,
  "humidity_percent": 65,
  "ambient_light":    400
}
```
`integration/decision_engine/engine.py` accepts this dict verbatim.
Updates reach the dashboard either via WebSocket or by polling `/sensors`.

---

## 8. Known limitations

1. **YOLOv8 (COCO)**
   - 8 of 10 BDD detection classes are covered by COCO; "traffic sign" vs
     "stop sign" is a partial match. Fine-tune on BDD for full coverage.

2. **Lane detection - classical**
   - Works on well-marked, daylight highway scenes. Will fail on sharp
     curves, occluded roads, very dark scenes. The DL mode addresses
     partially; neither is a production-grade system.

3. **Traffic sign GTSRB**
   - 43 German signs. Will mispredict on country-specific signs used
     outside Germany; extend by adding new classes & fine-tuning.

4. **Segmentation (SegFormer-B0 Cityscapes)**
   - Trained on German cities. Different countries can shift pixel-level
     accuracy. ACDC fine-tune (adverse weather) is the natural next step.

5. **No real-time temporal filtering**
   - The pipeline processes each frame independently. For dashboard SOTA,
     add a 3–5-frame hysteresis filter so warnings don't flicker.

6. **Decision engine is rule-based, not learned**
   - Thresholds in `integration/decision_engine/config.py` are sensible
     defaults. Production ADAS would learn decision boundaries.

7. **Synthetic vs real data**
   - Synthetic images passed through the pipeline only prove that the code
     works end-to-end. **No accuracy claims** (`PIPELINE VALIDATION ONLY`)
     are valid until you train and evaluate on real held-out data.

8. **Sandbox / batch timing**
   - Latency numbers in `evaluation_report.json` are wall-clock on this
     sandbox machine. Expect 2–5× faster on a real GPU laptop; ~5× slower
     on a Raspberry Pi + EdgeTPU.

---

## 9. Final integration sequence (to be done **after** you have tested
all 11 algorithms above)

```
ESP32 sensors -> Wi-Fi/MQTT -> fastapi /sensors  -> Decision engine
                                                            |
                                                            v
Webcam -> integration/webcam/main.py -> WebcamPipeline.step -> Decision engine
                                                            |    +
                                                            v    +
                                                results.json + warning list
                                                            |
                                                            v
                                            FastAPI /warnings -> frontend dashboard
```

The end-to-end GUI loop can be tested without the ESP32 by passing
fake ESP32 dicts (see `integration/decision_engine/demo.py`).
