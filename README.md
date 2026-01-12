# SfM 3D Reconstruction

Structure from Motion pipeline that reconstructs 3D point clouds from image sequences.

## What is 3D Reconstruction?

3D reconstruction is the process of recovering the three-dimensional structure of a scene from 2D images. By analyzing how objects appear from different viewpoints, we can estimate both the camera positions and the 3D geometry of the scene.

This technique is widely used in:
- **Robotics & Autonomous Vehicles**: Building maps for navigation and localization (SLAM)
- **AR/VR**: Creating 3D assets and spatial understanding
- **Surveying & Mapping**: Generating terrain models from drone/satellite imagery
- **Cultural Heritage**: Digitizing artifacts and historical sites
- **Film & VFX**: Creating 3D environments from footage

## How It Works

1. **Feature Detection**: SIFT keypoints extracted from each image with CLAHE preprocessing and non-maximum suppression
2. **Feature Matching**: FLANN-based matching with Lowe's ratio test across all image pairs
3. **Initial Pair Selection**: Finds the best starting pair by maximizing baseline (low homography/fundamental inlier ratio)
4. **Two-View Initialization**: Computes Essential matrix via 8-point RANSAC, recovers pose, triangulates initial points
5. **Incremental Registration**: Adds remaining cameras via PnP-RANSAC using 2D-3D correspondences
6. **Triangulation**: New 3D points triangulated as each camera is added
7. **Bundle Adjustment**: Joint optimization of all cameras and points using GTSAM

## Installation

```bash
pip install numpy opencv-python matplotlib plotly gtsam
```

## Usage

Place your images in a folder (e.g., `buddha/`) and run:

```bash
# Full pipeline with bundle adjustment
python bundle_adjustment.py

# Without bundle adjustment
python reconstruction_before_BA.py

# Visualize epipolar geometry
python visualize_epipolar.py
```

Update the camera intrinsics in `F_E_matrices.py` if needed:
```python
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])
```

## Output

Interactive 3D visualization via Plotly showing reconstructed points and camera positions.
