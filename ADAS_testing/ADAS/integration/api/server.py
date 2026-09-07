"""
server.py
=========
FastAPI backend for the ADAS system. Exposes:

  GET  /                  -> 303 redirect to /dashboard
  GET  /dashboard         -> new self-contained ADAS-testing frontend
  GET  /health            -> service + model-status JSON
  GET  /models            -> per-task model status + checkpoint availability
  GET  /comparison        -> the shared four-backbone comparison table
  POST /sensor            -> ESP32 readings (JSON)
  GET  /sensor            -> last cached sensor reading
  POST /detect            -> webcam frame -> DetectionPacket for the *driver-
                             assistance* React frontend (listens on the
                             'adas-detection' window event). Frontend UNTOUCHED.
  POST /analyze           -> uploaded IMAGE -> full per-task results + annotated
                             base64 image (this is what the new test frontend uses)
  POST /analyze_video     -> uploaded VIDEO -> annotated MP4 (served back) + stats
  GET  /results/<path>    -> serve annotated media written under results/

Run (hosts on localhost:7000):
    cd ADAS && uvicorn integration.api.server:app --host 127.0.0.1 --port 7000
    # or
    python integration/api/server.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]                # ADAS/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.api.ml_bridge import from_ai_and_sensors
from integration.api import ml_service
import integration.api.ml_service as M

_STATIC = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"

APP_DIR = "integration"

# Built merged ADAS frontend (one port, single app). The build can live either
# next to this repo (ADAS_testing/../ADAS_frontend) or beside the ADAS folder.
# We pick the first candidate that actually contains a production build; if none
# is found we fall back to the old self-contained dashboard.html.
FROM_SHARED_DIR = PROJECT_ROOT.parent
_FRONTEND_CANDIDATES = [
    PROJECT_ROOT.parent.parent / "ADAS_frontend" / "dist",   # <workspace>/ADAS_frontend/dist
    PROJECT_ROOT.parent / "ADAS_frontend" / "dist",          # ADAS_testing/ADAS_frontend/dist
    PROJECT_ROOT / "frontend_dist",                          # ADAS/frontend_dist
]
FRONTEND_DIST = next((p for p in _FRONTEND_CANDIDATES if (p / "index.html").exists()), None)

# ======================================================================= #
#  Sensor cache (ESP32 -> /sensor; dashboard -> /sensor)
# ======================================================================= #
LATEST_SENSOR = {
    "distance_cm": None, "rain": None, "temperature_c": None,
    "humidity_percent": None, "ambient_light": None,
    "speed_kph": 0, "gear": "D", "_ts": None,
}


class SensorPacket(BaseModel):
    distance_cm: float | None = None
    rain: bool | None = None
    temperature_c: float | None = None
    humidity_percent: float | None = None
    ambient_light: float | None = None
    speed_kph: float | None = None
    gear: str | None = None


app = FastAPI(title="ADAS Backend", version="2.0",
              description="ADAS tester: perception modules + transformer comparison.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

# Lazy model warm-er (no heavy import at import time).
_MODEL_TASKS = None
_WARM_DONE = False


async def _warm():
    global _MODEL_TASKS, _WARM_DONE
    await asyncio.sleep(0)
    if not _WARM_DONE:
        _WARM_DONE = True


# ======================================================================= #
#  Routes
# ======================================================================= #
# Serve the built SPA bundle if present (the single ADAS frontend on :7000).
if FRONTEND_DIST is not None and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
def _root():
    return _frontend_page()


@app.get("/dashboard")
def _dashboard():
    return _frontend_page()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ADAS backend",
        "frontend": "/dashboard",
        "models": ml_service.build_status(),
        "sensor_cache_ts": LATEST_SENSOR["_ts"],
        "ts": round(time.time(), 3),
    }


@app.get("/models")
def models_status():
    return ml_service.build_status()


@app.get("/comparison")
def comparison():
    return {"models": M.BACKBONES, "rows": M.load_comparison()}


# Non-blocking comparison runner (launches the training/eval subprocess in the
# background so the server stays responsive; results land in results/comparison.json).
_COMPARE_RUNNING = False
_COMPARE_NOTE = ""


@app.get("/comparison/status")
def comparison_status():
    return {"running": _COMPARE_RUNNING, "note": _COMPARE_NOTE}


@app.post("/comparison_run")
async def comparison_run():
    global _COMPARE_RUNNING, _COMPARE_NOTE
    if _COMPARE_RUNNING:
        return JSONResponse({"status": "running", "note": _COMPARE_NOTE,
                             "rows": M.load_comparison()})
    _COMPARE_RUNNING = True
    _COMPARE_NOTE = "Started."

    def _run():
        global _COMPARE_RUNNING, _COMPARE_NOTE
        try:
            cmd = [sys.executable, "-m", "algorithms._shared.run_comparison",
                   "--epochs", "30", "--pretrained", "--augment"]
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                  capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                _COMPARE_NOTE = "Run failed. See server logs. (Low-level: " + (proc.stderr or "")[-200:] + ")"
            else:
                _COMPARE_NOTE = "Run complete."
        except subprocess.TimeoutExpired:
            _COMPARE_NOTE = "Run timed out after 1 hour (still produced partial results?)."
        except Exception as e:
            _COMPARE_NOTE = f"Run error: {e}"
        finally:
            _COMPARE_RUNNING = False

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "note": _COMPARE_NOTE,
                         "rows": M.load_comparison()})


# ---- dataset demo samples (Day / Night / Fog / Rain) ----
DATASETS_DIR = PROJECT_ROOT / "datasets"
SAMPLE_DIRS = {"Day": 2, "Night": 2, "Fog": 2, "Rain": 2}
SAMPLE_EXTS = ("jpg", "jpeg", "png")


@app.get("/dataset/samples")
def dataset_samples(random: bool = False):
    """Return a few demo images per folder.

    With ``random=true`` (the frontend "Load fresh samples"/reload button) a
    DIFFERENT random selection is returned each call, so the demo gallery can be
    reshuffled live. Always avoid junk/corrupt files.
    """
    import random as _random
    samples = []
    if DATASETS_DIR.exists():
        for d, n in SAMPLE_DIRS.items():
            folder = DATASETS_DIR / d
            if not folder.is_dir():
                continue
            # collect real image files (skip macOS ._*, Thumbs.db, hidden, etc.)
            files = [p.name for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower().lstrip(".") in SAMPLE_EXTS
                     and not p.name.startswith("._")
                     and not p.name.lower().startswith("thumbs.db")
                     and not p.name.lower().startswith(".")]
            files.sort()
            if not files:
                continue
            if random:
                picked = _random.sample(files, min(n, len(files)))
            else:
                picked = files[:n]
            for f in picked:
                samples.append({"dir": d, "file": f, "url": f"/dataset/{d}/{f}"})
    return {"samples": samples}


@app.get("/dataset/{dir}/{file}")
def dataset_file(dir: str, file: str):
    root = DATASETS_DIR.resolve()
    target = (root / dir / file).resolve()
    if not target.is_file() or root not in target.parents:
        return JSONResponse({"error": "not found"}, status_code=404)
    suffix = target.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return FileResponse(str(target), media_type=media_type)


@app.post("/sensor")
async def post_sensor(payload: SensorPacket):
    data = payload.model_dump()
    data["_ts"] = round(time.time(), 3)
    LATEST_SENSOR.update({k: v for k, v in data.items() if k != "_ts"})
    return {"status": "ok", "_ts": data["_ts"]}


@app.get("/sensor")
def get_sensor():
    return LATEST_SENSOR


@app.post("/detect")
async def post_detect(file: UploadFile = File(...)):
    """Webcam frame -> DetectionPacket (for the untouched React frontend)."""
    raw = await file.read()
    bgr = _decode(raw)
    if bgr is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    h, w = bgr.shape[:2]

    # Lightweight run for the live bridge: weather + objects + lanes + signs.
    # (Segmentation is skipped in the live stream for latency.)
    ai_result = {
        "frame_id": 0, "timestamp": round(time.time(), 3),
    }
    try:
        weather = M.get("weather")
        if weather is not None:
            ai_result["weather"] = weather.predict(bgr)
    except Exception as e:
        ai_result["weather"] = {"status": "error", "error": str(e)}
    try:
        obj = M.get("objects")
        if obj is not None:
            ai_result["objects"] = obj.predict(bgr)
    except Exception as e:
        ai_result["objects"] = {"status": "error", "error": str(e)}
    try:
        lane = M.get("lanes")
        if lane is not None:
            ai_result["lanes"] = lane.predict(bgr)
    except Exception as e:
        ai_result["lanes"] = {"status": "error", "error": str(e)}
    try:
        sign = M.get("signs")
        if sign is not None:
            ai_result["signs"] = sign.predict(bgr)
    except Exception as e:
        ai_result["signs"] = {"status": "error", "error": str(e)}

    # `image=bgr` lets the bridge read the traffic-light lamp colour so the
    # UI can distinguish red / yellow / green instead of always assuming green.
    packet = from_ai_and_sensors(ai_results=ai_result, sensor_data=LATEST_SENSOR,
                                 img_w=w, img_h=h, image=bgr)
    return JSONResponse(_frontend_safe(packet))


@app.post("/step")
async def step(file: UploadFile = File(...)):
    """One full ADAS step: run the pipeline, combine with the cached sensor
    readings, and return the DecisionEngine warnings. Used by the tester to
    show the aggregated ADAS alerts alongside a frame."""
    raw = await file.read()
    bgr = _decode(raw)
    if bgr is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)

    res = M.analyze_image(bgr, save_overlay_path=None)
    from integration.decision_engine.engine import DecisionEngine

    # Turn ml_service results into the dict shape the DecisionEngine expects.
    ai = {
        "weather": res["tasks"].get("weather", {}),
        "objects": res["tasks"].get("objects", {}),
        "lanes":   res["tasks"].get("lanes", {}),
        "signs":   res["tasks"].get("signs", {}),
    }
    de = DecisionEngine()
    verdict = de.evaluate(ai, LATEST_SENSOR)

    ok, buf = cv2.imencode(".jpg", res["overlay_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return {
        "success": True,
        "inference_time_ms": res["inference_time_ms"],
        "verdict": verdict,
        "image_base64": base64.b64encode(buf.tobytes()).decode() if ok else None,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Uploaded image -> full perception report + annotated image (base64)."""
    raw = await file.read()
    bgr = _decode(raw)
    if bgr is None:
        return JSONResponse({"error": "could not decode image (jpg/png needed)"},
                            status_code=400)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = RESULTS_DIR / "integration" / f"analyze_{stamp}.jpg"
    res = M.analyze_image(bgr, save_overlay_path=save_path)

    # encode annotated image as base64 data URI for the browser
    ok, buf = cv2.imencode(".jpg", res["overlay_bgr"],
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    res["image_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    res["info"] = res["overlay"]["info"]
    res["inference_time_ms"] = res["inference_time_ms"]
    # strip internal BGR
    res.pop("overlay_bgr", None)
    return JSONResponse(res)


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), max_frames: int = 0):
    """Uploaded video -> runs the pipeline per frame, writes an annotated MP4
    and a sample annotated JPEG, returns stats + their URLs."""
    raw = await file.read()
    tmp = RESULTS_DIR / "integration" / f"_upload_{int(time.time())}_{file.filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)

    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        return JSONResponse({"error": "could not decode video"}, status_code=400)
    cap.release()

    stats, first_frame = M.process_video(tmp, max_frames=max_frames or None)

    # serve URLs
    vid = Path(stats["video_path"])
    sample = RESULTS_DIR / "integration" / f"{vid.stem}_sample.jpg"
    cv2.imwrite(str(sample), first_frame)
    ok, buf = cv2.imencode(".jpg", first_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    stats["sample_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
    stats["video_url"] = "/results/" + vid.relative_to(RESULTS_DIR).as_posix()
    stats["sample_url"] = "/results/" + sample.relative_to(RESULTS_DIR).as_posix()
    return JSONResponse(stats)


@app.get("/results/{path:path}")
def results(path: str):
    target = (RESULTS_DIR / path).resolve()
    # only allow files under results/
    if not str(target).startswith(str(RESULTS_DIR.resolve())):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not target.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    media_type = "image/jpeg"
    if target.suffix == ".mp4":
        media_type = "video/mp4"
    return FileResponse(str(target), media_type=media_type)


def _frontend_page() -> HTMLResponse:
    """Return the merged ADAS frontend.

    Preference order:
      1. dist/index_inline.html  — a single self-contained file with the CSS +
         JS inlined. Nothing to fetch, so it can never be a blank page.
      2. dist/index.html         — the normal SPA build (needs /assets served).
      3. dashboard_legacy.html   — the old self-contained tester page.
    """
    if FRONTEND_DIST is not None:
        inline = FRONTEND_DIST / "index_inline.html"
        if inline.exists():
            return HTMLResponse(inline.read_text(encoding="utf-8"))
        if (FRONTEND_DIST / "index.html").exists():
            # No inlined copy yet -> build one on the fly so the page is
            # always self-contained (never a blank page from /assets issues).
            try:
                _build_inline(FRONTEND_DIST)
                if inline.exists():
                    return HTMLResponse(inline.read_text(encoding="utf-8"))
            except Exception:
                pass
            return HTMLResponse((FRONTEND_DIST / "index.html").read_text(encoding="utf-8"))
    legacy = _STATIC / "dashboard_legacy.html"
    if legacy.exists():
        return HTMLResponse(legacy.read_text(encoding="utf-8"))
    return HTMLResponse((_STATIC / "dashboard.html").read_text(encoding="utf-8"))


@app.get("/adas_bridge.js", response_class=HTMLResponse)
def _bridge_js():
    return (_STATIC / "adas_bridge.js").read_text(encoding="utf-8")


# ======================================================================= #
#  helpers
# ======================================================================= #
def _decode(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _build_inline(dist: Path) -> None:
    """Write dist/index_inline.html as a single self-contained file (CSS + JS
    inlined) so the frontend never depends on /assets being reachable."""
    import re
    html = (dist / "index.html").read_text(encoding="utf-8")
    assets = dist / "assets"
    css = next((assets / f).read_text(encoding="utf-8")
               for f in sorted(p.name for p in assets.iterdir()) if f.endswith(".css"))
    js = next((assets / f).read_text(encoding="utf-8")
              for f in sorted(p.name for p in assets.iterdir()) if f.endswith(".js"))
    out = re.sub(r'<script type="module"[^>]*src="[^"]*"[^>]*></script>', '', html)
    out = re.sub(r'<link rel="stylesheet"[^>]*>', '', out)
    out = out.replace("</head>", f"  <style>\n{css}\n  </style>\n  </head>")
    out = out.replace("</body>", f'  <script type="module">\n{js}\n  </script>\n  </body>')
    (dist / "index_inline.html").write_text(out, encoding="utf-8")


def _frontend_safe(packet: dict) -> dict:
    """Keep everything the ARES display needs, and surface the weather class +
    confidence for the merged ADAS frontend (which shows a condition chip)."""
    keys_to_drop = {"_ts", "_signs"}
    out = {k: v for k, v in packet.items() if k not in keys_to_drop}
    # promote weather to public fields (the frontend reads weather / weatherConf)
    out["weather"] = packet.get("_weather")
    out["weatherConf"] = packet.get("_weather_conf")
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("integration.api.server:app", host="127.0.0.1", port=7000,
                reload=False, log_level="info")
