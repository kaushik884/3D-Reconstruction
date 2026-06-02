# Kaushik Deo
# April 23 2026
"""Image and camera-intrinsic loading."""
from __future__ import annotations
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

VALID_EXTS = (".png", ".jpg", ".jpeg")

def list_images(path: str) -> List[str]:
    return sorted([f for f in os.listdir(path) if f.lower().endswith(VALID_EXTS)])


def read_images(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Return (rgb, gray, filenames). Grayscale images are normalized to [0, 255]."""
    filenames = list_images(path)
    rgb, gray = [], []
    for f in filenames:
        # OpenCV loads BGR; convert to RGB for consistency with rest of the pipeline
        img = cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB)
        # Min-max normalize grayscale to [0, 255] so SIFT sees consistent contrast per image
        g = cv2.normalize(
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        rgb.append(img)
        gray.append(g)
    print(f"Loaded {len(rgb)} images from {path} (shape={rgb[0].shape})")
    return rgb, gray, filenames


def load_dtu_intrinsics(cams_dir: str, filenames: List[str]) -> List[np.ndarray]:
    """Load per-image 3x3 intrinsics from MVSNet-style cam.txt files."""
    Ks = []
    for f in filenames:
        base = os.path.splitext(f)[0]
        cam_path = os.path.join(cams_dir, f"{base}_cam.txt")
        K = _parse_mvsnet_cam(cam_path)
        Ks.append(K)
    return Ks


def _parse_mvsnet_cam(path: str) -> np.ndarray:
    """Parse MVSNet cam.txt format and return the 3x3 intrinsic matrix."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh.readlines()]

    # Find "intrinsic" header and read the next 3 lines
    idx = lines.index("intrinsic")
    rows = [list(map(float, lines[idx + 1 + i].split())) for i in range(3)]
    return np.array(rows, dtype=np.float64)


def resolve_intrinsics(
    cams_dir: Optional[str], filenames: List[str], fallback_K: Optional[np.ndarray]
) -> np.ndarray:
    """Return a single K matrix (averaged across cams if per-image, else the fallback)."""
    # Prefer per-image intrinsics from DTU cam files; average into a single K for the run
    if cams_dir is not None and os.path.isdir(cams_dir):
        Ks = load_dtu_intrinsics(cams_dir, filenames)
        K = np.mean(np.stack(Ks), axis=0)
        print(f"Loaded {len(Ks)} per-image intrinsics from {cams_dir}; using mean K.")
        return K
    if fallback_K is None:
        raise ValueError("No cams_dir and no fallback K provided.")
    return fallback_K.astype(np.float64)
