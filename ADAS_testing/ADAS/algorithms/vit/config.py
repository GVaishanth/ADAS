"""
config.py - ViT module.

All settings here are SHARED across ViT, Swin, ConvNeXt, and EfficientFormer
so the four-model comparison is fair. We just pick the specific timm model
name for ViT in model.py.
"""

from pathlib import Path

# Single source of truth for the class list + input size (Algorithm 1).
from algorithms.preprocessing.config import WEATHER_CLASSES, INPUT_SIZE

# ----- project paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR   = DATASETS_DIR / "bdd100k_weather"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_PREP  = MODELS_DIR / "preprocessing"  # not used here, but documents paths
MODEL_OUT    = MODELS_DIR / "vit"
RESULTS_OUT  = RESULTS_DIR / "vit"
COMPARISON_FILE = RESULTS_DIR / "comparison.json"

# Match Algorithm 1's preprocessing pipeline (see algorithms/preprocessing/config.py)
# (WEATHER_CLASSES + INPUT_SIZE now imported above.)
NUM_CLASSES = len(WEATHER_CLASSES)

# ---- ViT-specific ----
# CPU-friendly tiny tier for the sandbox comparison. Swap to
# "vit_base_patch16_224.augreg2_in21k_ft_in1k" (~86M) for the full-size run.
TIMM_MODEL = "vit_tiny_patch16_224"

# ---- training defaults ----
SEED            = 42
DEFAULT_EPOCHS  = 4
DEFAULT_BATCH   = 8
DEFAULT_LR      = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0            # raise to 2 / 4 on a stronger laptop

# ---- device / runtime ----
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
