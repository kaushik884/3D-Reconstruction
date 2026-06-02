# Kaushik Deo
# April 23 2026
"""Configuration for the 3D reconstruction pipeline. Dataset presets live here alongside the Config dataclass used by every stage.
To add a new dataset, append an entry to `DATASET_PRESETS`.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


DATASET_PRESETS = {
    "buddha": {
        "images_dir": "datasets/buddha",
        "cams_dir": None,
        # 1080x1920 portrait
        "K": np.array([[1200.0, 0.0, 540.0],
                       [0.0, 1200.0, 960.0],
                       [0.0, 0.0, 1.0]]),
        "name": "buddha",
    },
    "chair1": {
        "images_dir": "datasets/chair1",
        "cams_dir": None,
        # 1080x1920 portrait
        "K": np.array([[1200.0, 0.0, 540.0],
                       [0.0, 1200.0, 960.0],
                       [0.0, 0.0, 1.0]]),
        "name": "chair1",
    },
    "chair2": {
        "images_dir": "datasets/chair2",
        "cams_dir": None,
        # 1080x1920 portrait
        "K": np.array([[1200.0, 0.0, 540.0],
                       [0.0, 1200.0, 960.0],
                       [0.0, 0.0, 1.0]]),
        "name": "chair2",
    },
    "cup": {
        "images_dir": "datasets/cup",
        "cams_dir": None,
        # 1080x1920 portrait
        "K": np.array([[1200.0, 0.0, 540.0],
                       [0.0, 1200.0, 960.0],
                       [0.0, 0.0, 1.0]]),
        "name": "cup",
    },
    "skull": {
        "images_dir": "datasets/skull",
        "cams_dir": None,
        # 1600x1200 landscape
        "K": np.array([[1500.0, 0.0, 800.0],
                       [0.0, 1500.0, 600.0],
                       [0.0, 0.0, 1.0]]),
        "name": "skull",
    },
    "dtu_scan11": {
        "images_dir": "datasets/dtu/scan11/images",
        "cams_dir": "datasets/dtu/scan11/cams_1",
        "K": None,
        "name": "11",
    },
    "dtu_scan114": {
        "images_dir": "datasets/dtu/scan114/images",
        "cams_dir": "datasets/dtu/scan114/cams_1",
        "K": None,  # Loaded from cams_dir (MVSNet-style cam.txt files)
        "name": "114",
    },
    "dtu_scan118": {
        "images_dir": "datasets/dtu/scan118/images",
        "cams_dir": "datasets/dtu/scan118/cams_1",
        "K": None,  # Loaded from cams_dir (MVSNet-style cam.txt files)
        "name": "118",
    }
}


@dataclass
class Config:
    # Dataset / IO
    dataset: str = "dtu_scan33"
    images_dir: Optional[str] = None
    cams_dir: Optional[str] = None
    K: Optional[np.ndarray] = None
    name: Optional[str] = None
    output_dir: str = "outputs"

    # Stages to run. Subset of: epipolar, sparse, depth, dense, classify
    stages: List[str] = field(default_factory=lambda: ["sparse", "depth", "dense", "classify"])

    # Feature extraction
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    match_ratio: float = 0.8

    # Epipolar / RANSAC
    ransac_threshold_px: float = 1.0
    ransac_max_iters: int = 2000

    # Sparse SfM
    pnp_reprojection_error: float = 5.0
    triangulation_reproj_max: float = 5.0

    # Bundle adjustment
    run_ba: bool = True

    # Depth
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # Dense
    dense_depth_tol: float = 0.02
    dense_min_consistent_views: int = 3
    dense_max_depth: float = 100.0
    dense_cleanup: bool = True

    # Classification
    classify_use_rgb: bool = False
    classify_num_points: int = 1024

    # Runtime
    device: str = "cuda"
    show_viz: bool = False
    seed: int = 42

    def resolve(self) -> "Config":
        """Fill missing fields from dataset preset and absolutize paths."""
        if self.dataset in DATASET_PRESETS and self.images_dir is None:
            preset = DATASET_PRESETS[self.dataset]
            if self.images_dir is None:
                self.images_dir = preset["images_dir"]
            if self.cams_dir is None:
                self.cams_dir = preset["cams_dir"]
            if self.K is None and preset["K"] is not None:
                self.K = preset["K"].copy()
            if self.name is None:
                self.name = preset["name"]

        if self.name is None:
            self.name = self.dataset

        if self.images_dir is not None:
            self.images_dir = _abs(self.images_dir)
        if self.cams_dir is not None:
            self.cams_dir = _abs(self.cams_dir)
        self.output_dir = _abs(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        return self

    def run_dir(self) -> str:
        d = os.path.join(self.output_dir, self.name)
        os.makedirs(d, exist_ok=True)
        return d

    def depth_dir(self) -> str:
        d = os.path.join(self.run_dir(), "depth")
        os.makedirs(os.path.join(d, "raw_npy"), exist_ok=True)
        os.makedirs(os.path.join(d, "visualizations"), exist_ok=True)
        return d

    def dense_ply_path(self) -> str:
        return os.path.join(self.run_dir(), f"{self.name}_dense_cleaned.ply")

    def sparse_npz_path(self) -> str:
        return os.path.join(self.run_dir(), f"{self.name}_sparse.npz")
