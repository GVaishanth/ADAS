# ADAS

This is a college-level **Advanced Driver Assistance System (ADAS)** project.

The idea of the project is simple: take camera input from a road scene, run a few computer vision / deep learning modules on it, and show useful driving-related outputs like weather condition, detected objects, lanes, traffic signs, road segmentation, and warning messages.

It is a working prototype, not a production ADAS system.

---

## What this project does

This project combines:

- weather / road-scene classification
- object detection
- lane detection
- traffic-sign recognition
- semantic segmentation
- a webcam / video pipeline
- a small decision engine for warnings
- a frontend dashboard for testing

The main goal is to study how different vision models can be used together for driver assistance.

---

## Main models used

### Weather / scene classification
Four backbones are included and compared:

- **ViT**
- **Swin Transformer**
- **ConvNeXt**
- **EfficientFormer**

### Other perception modules

- **YOLOv8** for object detection
- **OpenCV-based lane detection**
- **ResNet18-based traffic sign classifier**
- **SegFormer-B0** for semantic segmentation

---

## Project structure

One important thing about this repo: the main code is **not directly in the root**.

```text
ADAS/
├── README.md
├── ADAS Scripts.txt
├── TASK_SUMMARY.md
├── ADAS_frontend/
├── ADAS_testing/
│   └── ADAS/
│       ├── algorithms/
│       ├── datasets/
│       ├── integration/
│       ├── results/
│       └── requirements.txt
└── driver-assistance/
```

### What these folders are for

- **`ADAS_testing/ADAS`** → main Python project
- **`ADAS_frontend`** → merged frontend used with the backend
- **`driver-assistance`** → older ARES-style frontend kept in the repo

If you want to run the actual system, most of the time you will work inside:

```bash
ADAS_testing/ADAS
```

---

## Current dataset in this repo

The dataset currently present in the repo is inside:

```text
ADAS_testing/ADAS/datasets/
```

It is arranged as 4 folders:

- `Day`
- `Night`
- `Fog`
- `Rain`

These are mapped by `datasets/prepare_current.py` as:

- `Day -> clear`
- `Night -> night`
- `Fog -> foggy`
- `Rain -> rainy`

According to the current checked-in dataset report:

- total usable images: **1999**
- training images: **1599**
- validation images: **200**
- test images: **200**

There is also one unreadable rain image that gets skipped while preparing the dataset.

---

## Current comparison results

These values are taken from the current file:

```text
ADAS_testing/ADAS/results/comparison.json
```

| Model | Accuracy | Precision | Recall | F1 | Latency | FPS | Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt | **98.5%** | **98.51%** | **98.5%** | **98.50%** | 75.637 ms | 13.22 | 106.19 MB |
| ViT | 95.0% | 95.41% | 95.0% | 94.93% | **17.832 ms** | **56.08** | **21.12 MB** |
| EfficientFormer | 95.0% | 95.0% | 95.0% | 95.0% | 31.727 ms | 31.52 | 43.69 MB |
| Swin | 74.5% | 82.60% | 74.5% | 69.66% | 101.368 ms | 9.87 | 105.05 MB |

Related files:

- `ADAS_testing/ADAS/results/comparison.json`
- `ADAS_testing/ADAS/results/comparison_chart.png`

> These are the current stored results in the repo. They should be treated as project results for this setup, not as a final industry benchmark.

---

## Simple system flow

```text
camera / image / video
        |
        v
   preprocessing
        |
        +--> weather classifier
        +--> object detector
        +--> lane detector
        +--> traffic-sign classifier
        +--> segmentation
        |
        v
     backend API
        |
        v
   dashboard / output
```

---

## Features available in the project

### 1. Weather classification
The system predicts the road/weather condition from the image.

Current classes used in this repo:

- clear
- night
- foggy
- rainy

### 2. Object detection
YOLOv8 is used to detect road-related objects such as:

- cars
- buses
- trucks
- motorcycles
- people
- traffic lights
- stop signs

### 3. Lane detection
The project includes a classical lane detection pipeline using OpenCV.

It uses edge detection, ROI masking, Hough transform, and lane fitting.

### 4. Traffic-sign recognition
Traffic signs are handled using a classifier built for GTSRB-style classes.

### 5. Semantic segmentation
SegFormer-B0 is used to segment the scene into classes like road, vehicle, person, sky, etc.

### 6. Video and webcam support
The backend can process:

- a single image
- a video file
- live webcam frames

### 7. Decision engine
A small rule-based decision engine is included to produce warnings like:

- obstacle ahead
- low visibility
- pedestrian detected
- vehicle proximity
- low light
- rain detected

---

## Frontend

The repo has two frontend-related folders.

### `ADAS_frontend`
This is the main frontend used with the backend.

It includes:

- **Live** view
- **Image** upload
- **Video** upload
- **Compare** view for model comparison

Important files:

- `ADAS_frontend/src/App.tsx`
- `ADAS_frontend/src/globals.css`
- `ADAS_frontend/make_inline.py`

### `driver-assistance`
This is the older frontend UI kept in the repository.

It listens for the browser event:

```text
adas-detection
```

The bridge file for that is:

- `ADAS_testing/ADAS/integration/api/adas_bridge.js`

---

## Backend API

The backend is in:

- `ADAS_testing/ADAS/integration/api/server.py`

Main routes:

| Route | Use |
|---|---|
| `GET /` | open the main frontend |
| `GET /dashboard` | same frontend entry |
| `GET /health` | backend health |
| `GET /models` | model loading status |
| `GET /comparison` | stored comparison results |
| `GET /dataset/samples` | sample dataset images |
| `POST /detect` | live frame detection |
| `POST /analyze` | image analysis |
| `POST /analyze_video` | video analysis |
| `POST /step` | pipeline + decision engine |
| `POST /sensor` / `GET /sensor` | sensor values |
| `GET /results/{path}` | saved output files |

---

## How to run

### 1. Go to the main project folder

From the repo root:

```bash
cd ADAS_testing/ADAS
```

### 2. Create a virtual environment

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset manifests

```bash
python datasets/prepare_current.py
```

### 5. Build the frontend

Go back to the repo root frontend folder:

```bash
cd ../../ADAS_frontend
npm install
npm run build
python make_inline.py
```

### 6. Start the backend

```bash
cd ../ADAS_testing/ADAS
venv/bin/python -m uvicorn integration.api.server:app --host 127.0.0.1 --port 7000
```

On Windows:

```powershell
venv\Scripts\python.exe -m uvicorn integration.api.server:app --host 127.0.0.1 --port 7000
```

### 7. Open in browser

```text
http://localhost:7000/
```

or

```text
http://localhost:7000/dashboard
```

---

## Running important modules separately

Run these from:

```bash
ADAS_testing/ADAS
```

### Re-run the four-model comparison

```bash
python datasets/prepare_current.py
python -m algorithms._shared.run_comparison --pretrained --epochs 30 --augment
python results/make_comparison_chart.py
```

### ViT

```bash
python algorithms/vit/train.py --pretrained --epochs 5
python algorithms/vit/evaluate.py
python algorithms/vit/predict.py --image path/to/image.jpg
```

### YOLOv8

```bash
python algorithms/yolo/predict.py --image path/to/image.jpg
python algorithms/yolo/predict.py --source 0
```

### Lane detection

```bash
python algorithms/lane_detection/predict.py --image path/to/image.jpg
python algorithms/lane_detection/predict.py --source 0
```

### Traffic-sign recognition

```bash
python algorithms/traffic_sign/train.py --auto-download --epochs 8
python algorithms/traffic_sign/evaluate.py --weights models/traffic_sign/best.pt
python algorithms/traffic_sign/predict.py --image path/to/image.jpg
```

### Segmentation

```bash
python algorithms/segmentation/predict.py --image path/to/image.jpg
python algorithms/segmentation/predict.py --source 0
```

### Webcam pipeline and decision engine

```bash
python integration/webcam/main.py --source 0
python integration/decision_engine/demo.py
```

---

## Sensor data format

The backend and decision engine use values like this:

```json
{
  "distance_cm": 240,
  "rain": false,
  "temperature_c": 22,
  "humidity_percent": 65,
  "ambient_light": 400
}
```

---

## Files worth reading first

If you want to understand the project quickly, start with these files:

1. `ADAS_testing/ADAS/integration/api/server.py`
2. `ADAS_testing/ADAS/integration/api/ml_service.py`
3. `ADAS_testing/ADAS/datasets/prepare_current.py`
4. `ADAS_testing/ADAS/algorithms/_shared/run_comparison.py`
5. `ADAS_frontend/src/App.tsx`

---

## Limitations

This is still a student project prototype, so a few practical limitations are expected:

- repo structure is a little messy because frontend and backend are in separate top-level folders
- trained `.pt` model files are not fully included by default
- some older docs in the repo describe earlier runs
- a few advanced parts are still scaffold-level compared to the main working pipeline
- this project is useful for demonstration and study, not for real driving use

---

## Final note

This project is mainly meant for:

- learning
- experimentation
- comparing models
- building a working ADAS prototype for a college project

It is best seen as a modular academic prototype that brings multiple vision tasks into one system.
