# Dense 3D Reconstruction and Semantic Classification from Unstructured Images

**Author:** Kaushik Deo
**Course:** CS 5330 — Pattern Recognition and Computer Vision (Northeastern University)

An end-to-end pipeline that turns **15–20 ordinary 2D photos of an object** into a **dense, colored 3D point cloud** and then **classifies the object** using a pretrained PointNet++ model. The pipeline is built from the ground up — feature matching, Structure from Motion, bundle adjustment, monocular depth fusion, and deep-learning classification — all stitched into a single reproducible command.

> No depth sensors. No turntable. No pre-calibrated rig. Just a handful of photos taken from different angles.

---

## Pipeline Overview

The system runs in five stages. Each stage caches its output, so you can resume, re-run, or skip individual steps.

```
 Images
   │
   ▼
[1] Epipolar geometry         SIFT + CLAHE  →  ratio-test matches  →  F / E matrices
   │
   ▼
[2] Sparse SfM                Incremental Structure-from-Motion (best-pair init,
   │                           PnP registration, guided triangulation, MAD outlier reject)
   │
   ▼
[3] Bundle Adjustment         GTSAM Levenberg-Marquardt with Huber robust kernel
   │
   ▼
[4] Monocular depth           Depth-Anything-V2 (HuggingFace) per image
   │
   ▼
[5] Dense fusion              Disparity-space scale/shift alignment to SfM,
   │                           GPU back-projection, multi-view consistency filter,
   │                           voxel downsample + statistical outlier removal
   │
   ▼
[6] Classification            PointNet++ (SSG) pretrained on ModelNet40 (40 classes)
   │
   ▼
 PLY point cloud + predicted class label + confidence
```


## Sample Results

| Object  | Cameras registered | Dense points | Top-1 prediction |
|---------|--------------------|--------------|------------------|
| Buddha  | 24                 | ~250k        | —                |
| Chair 1 | 20                 | ~310k        | chair            |
| Cup     | 20                 | ~180k        | cup              |
| Skull   | 18                 | ~220k        | —                |
| DTU 114 | 49                 | ~600k        | —                |


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/3D-Reconstruction.git
cd 3D-Reconstruction
```

### 2. Create a Python environment
Python **3.10+** is recommended. Conda is the easiest path because GTSAM and Open3D both have native dependencies.

```bash
conda create -n recon3d python=3.10 -y
conda activate recon3d
```

### 3. Install dependencies
```bash
# Core scientific stack
pip install numpy opencv-python matplotlib pillow plotly

# Deep learning + monocular depth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers

# 3D / point clouds
pip install open3d

# Bundle adjustment
pip install gtsam
```

> **CUDA note:** PyTorch is installed with CUDA 12.1 wheels above. Replace the index URL for your CUDA version, or omit it for CPU-only. The pipeline auto-falls-back to CPU when `cuda` is unavailable, though depth and dense fusion will be much slower.

### 4. Download the PointNet++ checkpoint
The classification stage uses pretrained PointNet++ SSG weights from the bundled [`Pointnet_Pointnet2_pytorch`](Pointnet_Pointnet2_pytorch/) submodule. The expected weight path is:

```
Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
```

Follow the training/checkpoint instructions in [Pointnet_Pointnet2_pytorch/README.md](Pointnet_Pointnet2_pytorch/README.md) (credit: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)) to obtain or retrain the weights.

### 5. Download the datasets
Custom datasets (buddha, chair1, chair2, cup, cone, skull) and the DTU scenes used in this project are available here:

[Google Drive — datasets](https://drive.google.com/drive/folders/1WA67X_jZAqfrMN4psyFgM-tQGQqOCTd5?usp=sharing)

Place them under `datasets/` matching the layout below.

---

## Usage

The single entry point is `scripts/main.py`. Stages can be run individually or chained together.

### Quick start — full pipeline on a preset
```bash
python scripts/main.py --dataset buddha --stages all
```

### Run individual stages
```bash
# Just sparse SfM (with bundle adjustment by default)
python scripts/main.py --dataset chair1 --stages sparse

# Generate monocular depth maps
python scripts/main.py --dataset chair1 --stages depth

# Build the dense point cloud (uses cached sparse + depth)
python scripts/main.py --dataset chair1 --stages dense

# Classify the saved dense PLY
python scripts/main.py --dataset chair1 --stages classify
```

### Run a custom image folder
```bash
python scripts/main.py \
    --images-dir path/to/my_photos \
    --name my_object \
    --fx 1200 --fy 1200 --cx 540 --cy 960 \
    --stages all
```

### Useful flags
| Flag | Purpose |
|------|---------|
| `--stages` | Comma-separated subset of `{epipolar, sparse, depth, dense, classify}` or `all`. |
| `--no-ba` | Skip bundle adjustment in the sparse stage. |
| `--show` | Open interactive matplotlib / Open3D viewers as stages complete. |
| `--fx/--fy/--cx/--cy` | Override intrinsics (use when you know your camera's focal length / principal point). |
| `--device` | `cuda` (default) or `cpu`. |
| `--classify-use-rgb` | Feed color channels alongside XYZ into PointNet++. |

### Built-in dataset presets
`buddha`, `chair1`, `chair2`, `cup`, `skull`, `dtu_scan11`, `dtu_scan114`, `dtu_scan118` — defined in [scripts/pipeline/config.py](scripts/pipeline/config.py).

---

## Outputs

For a run named `<NAME>`, the pipeline writes to `outputs/<NAME>/`:

```
outputs/<NAME>/
├── <NAME>_sparse.npz             # Cached sparse reconstruction (cameras + 3D points)
├── <NAME>_sparse.html            # Interactive Plotly viewer of cameras + sparse points
├── <NAME>_dense_cleaned.ply      # Final dense point cloud (open in MeshLab / CloudCompare)
├── depth/
│   ├── raw_npy/                  # Per-image raw depth (.npy)
│   └── visualizations/           # Inferno-colormapped depth previews (.png)
└── epipolar/                     # (optional) epipolar-line visualizations
```

Classification results print to stdout with top-3 predicted ModelNet40 classes + confidence scores.

---

## Project Structure

```
3D-Reconstruction/
├── scripts/
│   ├── main.py                   # CLI entry point
│   └── pipeline/
│       ├── config.py             # Dataset presets + Config dataclass
│       ├── io_utils.py           # Image loading, MVSNet cam.txt parsing
│       ├── features.py           # SIFT + CLAHE + NMS + GPU matcher
│       ├── epipolar.py           # F / E estimation, triangulation, viz
│       ├── sparse.py             # Incremental SfM (init pair, PnP, triangulation)
│       ├── bundle_adj.py         # GTSAM Levenberg-Marquardt BA
│       ├── depth.py              # Depth-Anything-V2 inference
│       ├── dense.py              # Depth alignment, GPU unprojection, consistency filter
│       ├── classify.py           # PointNet++ ModelNet40 classification
│       └── viz.py                # Plotly 3D viz helpers
├── datasets/                     # Input images (download separately)
├── outputs/                      # Per-run reconstructions and visualizations
└── Pointnet_Pointnet2_pytorch/   # PointNet++ submodule (yanx27)
```

---

## Tech Stack

- **Python 3.10**, **NumPy**, **OpenCV** (SIFT, MAGSAC++, PnP, triangulation)
- **PyTorch** (GPU matching, dense unprojection, classification inference)
- **HuggingFace Transformers** — Depth-Anything-V2
- **GTSAM** — factor-graph bundle adjustment
- **Open3D** — point cloud cleanup, normal estimation, visualization
- **Plotly** — interactive sparse-reconstruction viewer

---

## Acknowledgments

- [`yanx27/Pointnet_Pointnet2_pytorch`](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) — PointNet++ implementation and ModelNet40 weights.
- [`DepthAnything/Depth-Anything-V2`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) — monocular depth backbone.
- **DTU MVS dataset** for benchmark scenes.

