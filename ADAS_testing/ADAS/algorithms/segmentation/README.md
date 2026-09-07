# Algorithm 9 - Semantic Segmentation

HuggingFace **SegFormer-B0** pretrained on Cityscapes (`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`).

## Files
- `model.py`     SegmentationPredictor + Cityscapes class names/colormap.
- `train.py`     Optional fine-tune scaffold on Cityscapes (or ACDC).
- `evaluate.py`  mIoU / pixel-accuracy on Cityscapes val.
- `predict.py`   Image / video / webcam inference.

## Out-of-box behaviour
- Already produces a 19-class pixel map (road, sidewalk, building, ...).
- ~15 MB of weights downloaded on first construct.
- For our research direction ("road scene understanding under adverse
  weather"), the standard Cityscapes-trained model is the strongest
  starting point. For *fine-grained* fog/snow visibility, fine-tune on
  **ACDC** (Adverse Conditions Dataset) if you have it (~1 GB, 4
  conditions: fog, night, rain, snow).

## Dataset
- Cityscapes: https://www.cityscapes-dataset.com (registration required)
  - 5000 fine-labelled images, 19 classes.
  - Layout:
        cityscapes/leftImg8bit/{train,val,test}/<city>/<file>_leftImg8bit.png
        cityscapes/gtFine/{train,val,test}/<city>/<file>_gtFine_labelIds.png
  - Move train IDs via the official `createTrainIdLabelImgs.py`.
  - Storage: ~20 GB (images) + ~5 GB (fine labels).
- AC (alternative for adverse-weather research): ACDC (Sakaridis et al.)
  - 4 weather subsets, 1006 train / 508 val images each, ~1 GB.

## Routes
| Action | Command |
|---|---|
| Image | `python algorithms/segmentation/predict.py --image <path>.jpg` |
| Webcam | `python algorithms/segmentation/predict.py --source 0` |
| Evaluate on Cityscapes val | `python algorithms/segmentation/evaluate.py --cityscapes-root datasets/cityscapes --max-images 50` |

## API contract
```json
{
  "status": "success",
  "prediction": "scene_with_7_classes",
  "confidence": 0.14,
  "inference_time_ms": 81.2,
  "results": {
    "num_classes": 7,
    "classes": [
      {"class_id": 0, "class_name": "road",          "pixel_count": 12345, "percent": 32.5},
      {"class_id": 10,"class_name": "sky",           "pixel_count": 6789,  "percent": 17.8},
      ...
    ],
    "mask_path":    "results/segmentation/.../..._segmask.png",
    "overlay_path": "results/segmentation/.../..._seg.png",
    "model":        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "note":         "mIoU requires ground-truth masks - not computed here"
  }
}
```
