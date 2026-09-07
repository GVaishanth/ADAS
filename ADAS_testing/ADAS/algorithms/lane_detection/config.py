"""Lane-detection configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"

MODEL_OUT    = MODELS_DIR / "lane_detection"
RESULTS_OUT  = RESULTS_DIR / "lane_detection"
TUSIMPLE_DIR = DATASETS_DIR / "tusimple"

# Classical detector defaults - tuned for HD road footage / 720p
CANNY_LOW, CANNY_HIGH         = 40, 130
GAUSSIAN_KERNEL                = 5
HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 2, None, 70     # theta = np.pi/180
HOUGH_MIN_LINE, HOUGH_MAX_GAP  = 40, 300
NUM_PTS_PER_LANE               = 20                    # points sampled along the fitted curve

DEFAULT_LANE_COLOR_LEFT  = (0,   255, 0)   # green
DEFAULT_LANE_COLOR_RIGHT = (255, 0,   0)   # red
DEFAULT_ROI_VB_RATIO     = (0.45, 0.95)    # vertical top/bottom of the trapezoid ROI

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
