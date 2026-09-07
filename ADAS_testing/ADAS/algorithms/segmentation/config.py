"""Semantic segmentation configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"

MODEL_OUT    = MODELS_DIR / "segmentation"
RESULTS_OUT  = RESULTS_DIR / "segmentation"
CITYSCAPES_DIR = DATASETS_DIR / "cityscapes"

# Pretrained model on HuggingFace
HF_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DEFAULT_IMG_SIZE = 512         # resize shorter side before forward; small for laptop

SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH  = 8
DEFAULT_LR     = 5e-5

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# 19 Cityscapes evaluation classes (matches SegFormer's fine-tuning)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# A cityscape-style RGB palette for visualization (one entry per class)
CITYSCAPES_COLORMAP = [
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
]
