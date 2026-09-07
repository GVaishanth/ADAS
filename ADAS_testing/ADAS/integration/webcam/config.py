"""Integration / webcam configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODEL_OUT    = MODELS_DIR / "yolo"          # for default weights path
RESULTS_OUT  = RESULTS_DIR / "integration"

# Default pointer weights for each module
DEFAULT_VIT_WEIGHTS      = MODELS_DIR / "vit"         / "best.pt"
DEFAULT_SWIN_WEIGHTS     = MODELS_DIR / "swin"        / "best.pt"
DEFAULT_CONVNEXT_WEIGHTS = MODELS_DIR / "convnext"    / "best.pt"
DEFAULT_EFFFORMER_WEIGHTS= MODELS_DIR / "efficientformer"/"best.pt"
DEFAULT_YOLO_WEIGHTS     = "yolov8n.pt"
DEFAULT_LANE_WEIGHTS     = MODELS_DIR / "lane_detection" / "lane_unet.pt"
DEFAULT_SIGN_WEIGHTS     = MODELS_DIR / "traffic_sign" / "best.pt"


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
