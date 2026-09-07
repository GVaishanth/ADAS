"""Traffic sign (GTSRB) configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"

MODEL_OUT    = MODELS_DIR / "traffic_sign"
RESULTS_OUT  = RESULTS_DIR / "traffic_sign"
GTSRB_DIR    = DATASETS_DIR / "gtsrb"

# GTSRB has 43 classes
NUM_CLASSES = 43
IMG_SIZE = 64               # native GTSRB; resized for the model
TIMM_MODEL = "resnet18"      # small enough to train from scratch on a laptop

SEED = 42
DEFAULT_EPOCHS = 8
DEFAULT_BATCH  = 64
DEFAULT_LR     = 1e-3

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# Official 43 GTSRB class names (from the official `Readme`)
GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 t",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 t prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 t"
]
