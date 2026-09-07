# ADAS Frontend (merged, single app on port 7000)

This is the **merged** ADAS frontend — built **from the ARES frontend**
(`driver-assistance`, the automotive cabin UI) with ARES renamed to ADAS, the
`Automatic demo` button removed, the live camera kept in place, and the
**Image upload / Video upload / Model comparison** tooling added. It is served
by the FastAPI backend at **`http://localhost:7000`** (one frontend, one port).

It uses **same-origin relative URLs**, so it talks to the backend directly with
no CORS or localhost issues.

## Layout
```
ADAS_frontend/
├── index.html          entry page
├── package.json        react + vite
├── vite.config.js      builds a static, client-only SPA (base './')
├── src/
│   ├── main.tsx        mounts <App/>
│   ├── App.tsx         THE merged component (ARES styling + tabs + uploads)
│   └── globals.css     exact ARES stylesheet + a few additions (tabs/panels)
└── dist/               production build the backend serves (regenerate below)
```

## Build (produces `dist/`, which the backend serves)
```bash
npm install
npm run build
```
The backend (`integration/api/server.py`) automatically serves `dist/` at `/`
and `/dashboard` (falling back to the legacy tester page if the build is absent).

## Tabs
- **Live** (default) — ARES camera view showing real model detections + a
  weather/condition chip. `Automatic demo` is removed.
- **Image** — upload a road image → annotated image + per-module report.
- **Video** — upload a driving clip → annotated MP4 + stats + sample frame.
- **Compare** — the four-backbone Task-1 comparison table.
