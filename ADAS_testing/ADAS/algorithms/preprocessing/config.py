"""
config.py
=========
Central configuration for the Data Processing / Preprocessing module (Algorithm 1).

WHY ONE CONFIG FILE?
All four transformer models (ViT, Swin, ConvNeXt, EfficientFormer) must consume
the *same* preprocessed data and the *same* normalization, otherwise the final
model-comparison table would not be fair. Every setting that matters (paths,
image size, ImageNet stats, class names, split sizes) lives here once, and every
later module imports it from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout (see ROADMAP.md)
#   ADAS/
#     algorithms/preprocessing/   <- this module
#     datasets/                  <- raw + cleaned datasets
#     models/                    <- trained weights
#     results/                   <- outputs (reports, images, videos)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # the ADAS/ folder
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

BDD100K_ROOT = DATASETS_DIR / "bdd100k"              # where YOU put the official download
OUTPUT_DIR   = DATASETS_DIR / "bdd100k_weather"      # cleaned dataset we generate
RESULTS_PREP = RESULTS_DIR / "preprocessing"         # reports / plots / videos of this module

# ---------------------------------------------------------------------------
# BDD100K raw layout (verified against the official docs + download page)
#
#   bdd100k/
#   ├── images/
#   │   └── 100k/
#   │       ├── train/     70,000 images
#   │       ├── val/       10,000 images
#   │       └── test/      20,000 images  (labels NOT released -> we don't use it)
#   └── labels/
#       ├── bdd100k_labels_images_train.json
#       └── bdd100k_labels_images_val.json
#
# NOTE: BDD100K only publishes annotations for train and val. The official test
# set has no public labels (held out for the benchmark), so our own test split
# is carved out of the official val set, disjoint from our val set.
# ---------------------------------------------------------------------------
BDD_IMAGES_DIR   = BDD100K_ROOT / "images" / "100k"
BDD_LABELS_TRAIN = BDD100K_ROOT / "labels" / "bdd100k_labels_images_train.json"
BDD_LABELS_VAL   = BDD100K_ROOT / "labels" / "bdd100k_labels_images_val.json"
BDD_SPLIT_IMAGES = {
    "train": BDD_IMAGES_DIR / "train",
    "val":   BDD_IMAGES_DIR / "val",
    "test":  BDD_IMAGES_DIR / "test",   # reference only; labels unavailable
}

# ---------------------------------------------------------------------------
# Task: driving-condition classification.
# The dataset that actually ships in this repo is the four gathered folders
# {Day, Night, Fog, Rain} (see datasets/prepare_current.py). We map them onto
# these stable condition labels so the whole pipeline works out-of-the-box.
# (The original 6-class BDD100K weather vocabulary - clear / partly cloudy /
#  overcast / rainy / snowy / foggy - can be restored if you download BDD100K
#  and run the original datasets/build_dataset.py path instead.)
# ---------------------------------------------------------------------------
WEATHER_CLASSES = ["clear", "night", "foggy", "rainy"]

# ---------------------------------------------------------------------------
# Shared input configuration for ALL 4 transformer models.
# ViT / Swin-T / ConvNeXt-T / EfficientFormer all use 224x224 input and the
# standard ImageNet normalization expected by their pretrained checkpoints.
# ---------------------------------------------------------------------------
INPUT_SIZE     = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Prototype subset sizes (per class). Start small to validate the pipeline,
# then raise them when you train the real models.
# ---------------------------------------------------------------------------
SEED            = 42
TRAIN_PER_CLASS = 400   # train images per weather class
VAL_PER_CLASS   = 100   # val images per class (from official val)
TEST_PER_CLASS  = 100   # test images per class (from official val, disjoint)

# ---------------------------------------------------------------------------
# Sanity-check tolerances used by evaluate.py
# The check exists to catch pipeline errors (e.g. forgotten normalization ->
# mean ~0.5, std ~0.3), NOT to reject real adverse-weather photos, which
# legitimately have shifted statistics (foggy/snowy frames are grayish/bright).
# On real BDD100K photos expect mean ~0.00 +/- 0.05 and std ~1.00 +/- 0.10.
# ---------------------------------------------------------------------------
TENSOR_MEAN_TOL = 0.15
TENSOR_STD_TOL  = 0.25
