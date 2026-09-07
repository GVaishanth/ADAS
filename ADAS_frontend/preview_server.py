"""Lightweight preview server: serves the merged ADAS frontend and stubs the
model endpoints so the React app renders and is fully interactive with no ML
stack installed. Swap for the real backend in ADAS_testing/ADAS for live
inference (its routes are identical)."""
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

# real dataset images live in the ADAS project
DATASETS = HERE.parent / "ADAS_testing" / "ADAS" / "datasets"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

# serve dataset sample thumbnails via an explicit route (a StaticFiles mount at
# /dataset would shadow the /dataset/samples route below)
SAMPLE_DIRS = {"Day": 2, "Night": 2, "Fog": 2, "Rain": 2}


@app.get("/dataset/samples")
def dataset_samples():
    samples = []
    if DATASETS.exists():
        for d, n in SAMPLE_DIRS.items():
            folder = DATASETS / d
            if not folder.is_dir():
                continue
            files = sorted(p.name for p in folder.glob("*.jpg"))[:n]
            for f in files:
                samples.append({"dir": d, "file": f, "url": f"/dataset/{d}/{f}"})
    return {"samples": samples}


@app.get("/dataset/{dir}/{file}")
def dataset_file(dir: str, file: str):
    # resolve safely against the datasets root
    root = DATASETS.resolve()
    target = (root / dir / file).resolve()
    if not target.is_file() or root not in target.parents:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(target)


def frontend_page() -> HTMLResponse:
    for c in (DIST / "index_inline.html", DIST / "index.html"):
        if c.exists():
            return HTMLResponse(content=c.read_text("utf-8"), media_type="text/html")
    return HTMLResponse("<h1>build the frontend first: npm run build</h1>", status_code=503)


@app.get("/")
def root():
    return frontend_page()


@app.get("/dashboard")
def dashboard():
    return frontend_page()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    await file.read()
    return {
        "source": "model",
        "inference_time_ms": 61,
        "vehicle": {"detected": True, "distanceM": round(5.4, 1), "position": "front"},
        "rain": False, "lane": "steady", "trafficLight": "green", "speedLimit": 40,
        "roadSurface": "dry", "gear": "D", "speedKph": 62,
        "objects": [
            {"label": "car", "x": 30, "y": 40, "width": 22, "height": 26, "confidence": 0.93, "ttc_s": 3.2},
            {"label": "person", "x": 72, "y": 55, "width": 8, "height": 18, "confidence": 0.76, "ttc_s": 5.8},
        ],
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    await file.read()
    return {
        "image_base64": "",
        "inference_time_ms": 214,
        "info": {"weather": "Clear", "objects": 2, "lanes": 2, "seg_classes": 3},
        "tasks": {
            "weather": {"status": "ok", "prediction": "Clear", "confidence": 0.92,
                        "results": {"top_k": [{"class": "Clear", "confidence": 0.92},
                                               {"class": "Cloudy", "confidence": 0.05}]}},
            "objects": {"status": "ok", "results": {"detections": [
                {"class": "car", "confidence": 0.9}, {"class": "person", "confidence": 0.8}]}},
            "lanes": {"status": "ok", "results": {"num_lanes": 2, "mode": "classical"}},
            "signs": {"status": "ok", "prediction": "speed_40", "confidence": 0.88,
                      "results": {"top_k": [{"class_name": "speed_40"}]}},
            "seg": {"status": "ok", "results": {"num_classes": 3, "classes": [
                {"class_name": "road", "percent": 55}, {"class_name": "vehicle", "percent": 20}]}},
        },
    }


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), max_frames: int = 120):
    await file.read()
    return {
        "frames": min(max_frames, 120), "avg_ms": 180, "object_frames": 88,
        "weather_counts": {"Clear": 60, "Cloudy": 40},
        "sample_base64": "", "video_url": None,
    }


def comparison_rows():
    return [
        {"model": "vit", "accuracy": 0.9167, "precision": 0.9375, "recall": 0.9167,
         "f1": 0.9143, "inference_time_ms": 30, "fps": 33, "model_size_mb": 21},
        {"model": "efficientformer", "accuracy": 0.9167, "precision": 0.9375,
         "recall": 0.9167, "f1": 0.9143, "inference_time_ms": 35, "fps": 28, "model_size_mb": 44},
        {"model": "swin", "accuracy": 0.5, "precision": 0.3333, "recall": 0.5,
         "f1": 0.375, "inference_time_ms": 116, "fps": 9, "model_size_mb": 105},
        {"model": "convnext", "accuracy": 0.5, "precision": 0.3333, "recall": 0.5,
         "f1": 0.375, "inference_time_ms": 81, "fps": 12, "model_size_mb": 106},
    ]


@app.get("/comparison")
def comparison():
    return {"models": ["vit", "swin", "convnext", "efficientformer"], "rows": comparison_rows()}


@app.post("/comparison_run")
async def comparison_run():
    # In the real backend this kicks off the training/eval subprocess. Here we
    # simulate a finished run so the UI + button flow is verifiable.
    time.sleep(1.2)
    return {"status": "complete", "rows": comparison_rows(),
            "note": "Run complete (simulated in preview)."}
