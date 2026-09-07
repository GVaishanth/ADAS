# ADAS — Implementation Roadmap

**Project:** AI-Based Driver Assistance System using Vision Transformers and ESP32
**Research direction:** Road scene understanding under adverse weather conditions using modern Vision Transformer architectures, followed by ADAS perception modules.

## Phasing
Dataset → Algorithms (one at a time) → Testing → FastAPI → ESP32 communication → Frontend Dashboard → Final Integration

## Module Roadmap

| # | Module | Recommended Dataset | Est. Time | Role in ADAS |
|---|--------|---------------------|-----------|--------------|
| 1 | Data Processing / Preprocessing | BDD100K (diverse weather) + a few adverse-weather samples | 2–3 days | Resize/normalize/denoise/dehaze; robust input pipeline for all models |
| 2 | Vision Transformer (ViT) | BDD100K weather-condition subset (6 classes) | 3–4 days | Weather/scene classification — comparative model #1 |
| 3 | Swin Transformer | BDD100K weather subset (same split) | 2–3 days | Weather/scene classification — comparative model #2 |
| 4 | ConvNeXt | BDD100K weather subset (same split) | 2–3 days | Weather/scene classification — comparative model #3 |
| 5 | EfficientFormer | BDD100K weather subset (same split) | 2–3 days | Lightweight weather/scene classification — comparative model #4 |
| 6 | YOLOv8 Object Detection | BDD100K detection subset (10 classes) | 2–3 days | Detect cars, pedestrians, signs, traffic lights |
| 7 | Lane Detection | TuSimple (prototype) / CULane (advanced) | 3–4 days | Detect lane markings on the road |
| 8 | Traffic Sign Recognition | GTSRB (43 classes, ~50K images) | 2–3 days | Recognize traffic signs |
| 9 | Semantic Segmentation | Cityscapes (19 classes) / CamVid (lighter) / ACDC (adverse weather) | 3–4 days | Pixel-level road scene parsing |
| 10 | OpenCV + PyTorch Integration | Live webcam / a test driving video | 2–3 days | Real-time feed → preprocessing → unified model pipeline |
| 11 | Final Decision Engine / Aggregation | Sample model outputs + ESP32 sensor simulator | 3–4 days | Fuse all outputs + sensor data → ADAS warnings (JSON) |

## Key Notes
- **Modules 2–5 share the same BDD100K weather-condition split** so the final comparison table (Accuracy / Precision / Recall / F1 / Inference Time / FPS / Model Size) is fair and never fabricated.
- Full per-module dataset details (labels, classes, storage size, train/val/test split) are provided when we build each algorithm.
- The laptop runs all heavy AI; the ESP32 only reads sensors and receives decisions.
- Approximate total: ~5–6 weeks of part-time student work.

## Build Order
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

> I will not start the next module until you confirm the current one works and tell me to proceed.
