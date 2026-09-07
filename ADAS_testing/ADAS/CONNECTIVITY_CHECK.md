# Frontend ↔ Backend Connectivity Check + Hosting (localhost:7000)

## Hosting
- Backend hosts on **`http://localhost:7000`** (bound to **127.0.0.1** only — NOT 0.0.0.0).
- Verified at runtime: `Uvicorn running on http://127.0.0.1:7000`.

## How to run (your machine)
```powershell
cd ...\ADAS_testing\ADAS
python -m uvicorn integration.api.server:app --host 127.0.0.1 --port 7000
# or just:
python integration/api/server.py        (already defaults to 127.0.0.1:7000)
```
Open **http://localhost:7000**

## Frontend → Backend wiring (every call maps to a real route)
| Frontend call | Backend route | Purpose |
|---|---|---|
| `fetch('/detect')` | POST `/detect` | Live camera frame → DetectionPacket |
| `fetch('/analyze')` | POST `/analyze` | Image upload → per-task results |
| `fetch('/analyze_video')` | POST `/analyze_video` | Video upload → stats + annotated clip |
| `fetch('/comparison')` | GET `/comparison` | Load stored comparison table |
| `fetch('/comparison_run')` | POST `/comparison_run` | Kick off the four-backbone run |
| `fetch('/dataset/samples')` | GET `/dataset/samples` | Demo sample list (Day/Night/Fog/Rain) |
| `<img src="/dataset/{dir}/{file}">` | GET `/dataset/{dir}/{file}` | Sample thumbnails / previews |

All calls use **relative URLs** (no absolute `localhost:8000`), so there are no CORS or host issues.

## Live audit — all routes 200 on localhost:7000
```
GET  /                 200   frontend (self-contained, 0 external refs)
GET  /dashboard        200   same frontend
POST /detect           200   live camera
POST /analyze          200   image analyse
POST /analyze_video    200   video analyse
GET  /comparison       200   compare load
POST /comparison_run   200   compare run
GET  /dataset/samples  200   samples list
GET  /dataset/Day/...  200   sample image
GET  /health           200   health
```

## Note
The in-sandbox run above used a stub-free real backend import path; the real
perception endpoints need the ML stack (`pip install -r ADAS/requirements.txt`)
installed on your machine to produce live inference. The HTTP wiring, routes,
and frontend are identical.
