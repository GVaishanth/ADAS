"""Algorithm 1 — Data Processing / Preprocessing.

Public API (used by all later modules and by FastAPI):
    PreprocessingPipeline   -> JSON-friendly preprocess()
    BDDWeatherDataset       -> torch Dataset built from our manifest CSVs
    get_train_transforms()  -> augmentation pipeline (training)
    get_val_transforms()    -> deterministic pipeline (val/test/inference)
    apply_enhancement()     -> optional CLAHE / denoising
    load_manifest()         -> read one split CSV
"""

from .preprocessing import (
    BDDWeatherDataset,
    PreprocessingPipeline,
    apply_enhancement,
    get_train_transforms,
    get_val_transforms,
    load_manifest,
    tensor_to_image,
)

__all__ = [
    "BDDWeatherDataset",
    "PreprocessingPipeline",
    "apply_enhancement",
    "get_train_transforms",
    "get_val_transforms",
    "load_manifest",
    "tensor_to_image",
]
