"""ConvNeXt configuration."""

from pathlib import Path

# Single source of truth for the class list + input size (Algorithm 1).
from algorithms.preprocessing.config import WEATHER_CLASSES, INPUT_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODEL_OUT    = MODELS_DIR / "convnext"
RESULTS_OUT  = RESULTS_DIR / "convnext"
COMPARISON_FILE = RESULTS_DIR / "comparison.json"

# (WEATHER_CLASSES + INPUT_SIZE imported above.)
NUM_CLASSES = len(WEATHER_CLASSES)

# ConvNeXt-Tiny, the modernized ConvNet used as a CNN baseline.
TIMM_MODEL = "convnext_tiny.fb_in22k_ft_in1k"

SEED            = 42
DEFAULT_EPOCHS  = 4
DEFAULT_BATCH   = 8
DEFAULT_LR      = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 0

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
