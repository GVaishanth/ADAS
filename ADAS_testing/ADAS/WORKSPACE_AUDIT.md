# Workspace audit — what is kept, what is deleted, and why

The workspace was over its storage/file-count limit because it contained large,
**regenerable build artefacts**. Everything needed for the project is kept;
everything that can be rebuilt with a command has been removed so the repo stays
small and self-contained.

## Kept (all required files — the actual project)
| Area | Contents | Why it's kept |
|------|----------|---------------|
| `algorithms/` | All 11 modules (vit, swin, convnext, efficientformer, yolo, lane_detection, traffic_sign, segmentation, preprocessing, `_shared` helpers, `run_comparison.py`) | The Task-1 + Task-2 logic |
| `integration/` | `api/server.py`, `api/ml_service.py`, `api/ml_bridge.py`, `api/dashboard.html`, `api/adas_bridge.js`, decision_engine, webcam | The demo backend + new test frontend |
| `datasets/` | `Day/` `Night/` `Fog/` `Rain/` (80 images) + `bdd100k_weather/` manifests + `prepare_current.py` | The real input data + adapters |
| `results/` | `comparison.json`, `comparison_chart.png`, per-model `metrics.json`, `train_log.json`, `confusion_matrix.png`, `make_comparison_chart.py` | **The Task-1 deliverables** (numbers + charts) |
| `requirements.txt`, README/ROADMAP, `setup_env.sh`, `WORKSPACE_AUDIT.md` | Docs + installer | Reproduce everything |

## Deleted (regenerable / non-deliverable)
| Item | Why deleted | How to get it back |
|------|-------------|---------------------|
| `venv/` (≈104 MB) | The Python env — install fodder, not part of the repo | `bash setup_env.sh` |
| `driver-assistance/node_modules/` | npm build output | `npm install` |
| `driver-assistance/.npm` `.cache` `.config` `.wrangler` `.next` `.vinext` | build caches | regenerated automatically |
| `ADAS/models/*.pt` (trained weights) | large binaries; trimmed by the snapshot | `bash setup_env.sh --with-training` (retrains) |
| `yolov8n.pt` (6 MB) | auto-downloads | first YOLO run downloads it |
| `*.log`, `__pycache__/*`, `results/_sanity_*.png`, scratch annotated pngs/videos | debug/cache junk | not needed |
| `/tmp/node-v22*` | Node binary pulled in to run the frontend | install Node ≥22 yourself |

## Results preserved (Task 1 evidence) — `results/comparison.json`
| Model | Acc | Prec | Rec | F1 | Latency | FPS | Size |
|-------|-----|------|-----|-----|---------|-----|------|
| ViT | 0.9167 | 0.9375 | 0.9167 | 0.9143 | 30.1 ms | 33.2 | 21.1 MB |
| EfficientFormer | 0.9167 | 0.9375 | 0.9167 | 0.9143 | 35.4 ms | 28.3 | 43.7 MB |
| Swin | 0.5 | 0.3333 | 0.5 | 0.375 | 115.9 ms | 8.6 | 105.0 MB |
| ConvNeXt | 0.5 | 0.3333 | 0.5 | 0.375 | 81.5 ms | 12.3 | 106.2 MB |

## Final size
**13 MB · 211 files** (down from 119 MB · 10,000 files).
