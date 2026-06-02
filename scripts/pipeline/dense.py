# Kaushik Deo
# April 23 2026
"""Dense reconstruction: align monocular depth to SfM scale, unproject, consistency-filter. Unprojection + consistency runs on GPU via PyTorch.
"""
from __future__ import annotations
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np
import open3d as o3d
import torch


def align_depth_maps(
    reconstruction: Dict,
    depth_dir: str,
    filenames: Sequence[str],
) -> Dict[int, np.ndarray]:
    """Scale+shift each raw depth map to match sparse SfM depths via least squares."""
    aligned = {}
    raw_dir = os.path.join(depth_dir, "raw_npy")

    for cam_idx, cam in reconstruction["cameras"].items():
        img_name = filenames[cam_idx]
        base = os.path.splitext(img_name)[0]
        npy_path = os.path.join(raw_dir, f"{base}_depth.npy")
        if not os.path.exists(npy_path):
            print(f"[Dense] Warning: depth map missing for {img_name}")
            continue

        raw = np.load(npy_path)
        R, t = cam["R"], cam["t"]
        # Collect (mono_depth, sfm_depth) pairs at every SfM keypoint in this view
        mono, metric = [], []
        for pt_3d_idx, pt_2d in reconstruction["observations"].get(cam_idx, {}).items():
            pt_3d = np.array(reconstruction["points_3d"][pt_3d_idx])
            # Transform world point into this camera's frame to get its true z-depth
            pt_cam = R @ pt_3d.reshape(3, 1) + t
            z = pt_cam[2, 0]
            if z <= 0:
                continue
            u, v = int(round(pt_2d[0])), int(round(pt_2d[1]))
            if 0 <= v < raw.shape[0] and 0 <= u < raw.shape[1]:
                mono.append(raw[v, u])
                metric.append(z)

        if len(mono) < 2:
            print(f"[Dense] Camera {cam_idx}: not enough sparse anchors ({len(mono)})")
            continue
        A = np.vstack([mono, np.ones(len(mono))]).T
        # 1. Fit against inverse metric depth (disparity)
        b = 1.0 / np.array(metric) 
        (scale, shift), *_ = np.linalg.lstsq(A, b, rcond=None)
        
        # 2. Apply scale and shift to the raw disparity map
        aligned_disp = scale * raw + shift
        
        # 3. Prevent divide-by-zero for pixels that get pushed into negative disparity
        aligned_disp[aligned_disp <= 1e-4] = 1e-4
        
        # 4. Invert back to metric depth!
        dmap = 1.0 / aligned_disp
        
        aligned[cam_idx] = dmap

        print(f"[Dense] Cam {cam_idx} [{img_name}]: scale={scale:.4f} shift={shift:.4f} "
              f"(n_anchors={len(mono)})")
    return aligned


def unproject_with_consistency(
    aligned_depths: Dict[int, np.ndarray],
    reconstruction: Dict,
    K: np.ndarray,
    rgb_images: Sequence[np.ndarray],
    max_depth: float = 20.0,
    depth_tol: float = 0.01,
    min_consistent_views: int = 4,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Dense] Unprojection device: {dev}")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam_indices = sorted(aligned_depths.keys())

    # Pre-move neighbor tensors once to avoid repeated host->device transfers
    depth_tensors = {i: torch.as_tensor(aligned_depths[i], device=dev, dtype=torch.float32)
                     for i in cam_indices}
    R_tensors = {i: torch.as_tensor(reconstruction["cameras"][i]["R"], device=dev, dtype=torch.float32)
                 for i in cam_indices}
    t_tensors = {i: torch.as_tensor(reconstruction["cameras"][i]["t"], device=dev, dtype=torch.float32)
                 for i in cam_indices}

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for i, cam_idx in enumerate(cam_indices):
        depth = depth_tensors[cam_idx]
        R_ref = R_tensors[cam_idx]
        t_ref = t_tensors[cam_idx]
        h, w = depth.shape

        v_grid, u_grid = torch.meshgrid(
            torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij"
        )
        u = u_grid.flatten().float()
        v = v_grid.flatten().float()
        z = depth.flatten()

        # Drop invalid / far-away pixels before doing expensive math
        mask = (z > 0.1) & (z < max_depth)
        u, v, z = u[mask], v[mask], z[mask]
        if z.numel() == 0:
            continue

        # Back-project pixels into reference camera, then into world coordinates
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        p_cam = torch.stack((x_cam, y_cam, z))                     # (3, N)
        p_world = R_ref.T @ (p_cam - t_ref)                        # (3, N)

        # Check each point against up to 4 temporally adjacent views
        neighbors = [cam_indices[j] for j in range(max(0, i - 2), min(len(cam_indices), i + 3))
                     if cam_indices[j] != cam_idx]
        count = torch.zeros(p_world.shape[1], dtype=torch.int32, device=dev)

        for nb in neighbors:
            # Project into the neighbor view and compare projected vs stored depth
            p_nb = R_tensors[nb] @ p_world + t_tensors[nb]         # (3, N)
            z_nb = p_nb[2]
            front = z_nb > 0.1
            if not front.any():
                continue
            u_nb = (fx * p_nb[0, front] / z_nb[front] + cx).long()
            v_nb = (fy * p_nb[1, front] / z_nb[front] + cy).long()
            in_bounds = (u_nb >= 0) & (u_nb < w) & (v_nb >= 0) & (v_nb < h)
            front_idx = torch.where(front)[0][in_bounds]
            map_d = depth_tensors[nb][v_nb[in_bounds], u_nb[in_bounds]]
            proj_d = z_nb[front_idx]
            # Relative depth error < tolerance means the neighbor agrees
            consistent = torch.abs(map_d - proj_d) / proj_d < depth_tol
            count[front_idx[consistent]] += 1

        # Keep only points validated by enough neighboring views (geometric consistency)
        keep = count >= (min_consistent_views - 1)
        if not keep.any():
            continue
        pts = p_world[:, keep].T.cpu().numpy()

        u_orig = u[keep].long().cpu().numpy()
        v_orig = v[keep].long().cpu().numpy()
        colors = rgb_images[cam_idx][v_orig, u_orig]
        all_points.append(pts)
        all_colors.append(colors)
        print(f"[Dense] Cam {cam_idx}: kept {int(keep.sum())}/{z.numel()} pts")

    if not all_points:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(all_points), np.vstack(all_colors)


def build_and_save_point_cloud(
    points_3d: np.ndarray,
    colors: np.ndarray,
    output_path: str,
    cleanup: bool = True,
    show: bool = False,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
    norm_colors = colors.astype(np.float64) / (255.0 if colors.max() > 1.0 else 1.0)
    pcd.colors = o3d.utility.Vector3dVector(norm_colors)

    # Optional cleanup: voxel downsample for uniform density + remove statistical outliers
    if cleanup and len(pcd.points) > 20:
        bbox = pcd.get_max_bound() - pcd.get_min_bound()
        # Divide the longest dimension into 500 grid cells. 
        # Increase 500 for more detail, decrease for smaller file sizes.
        voxel_size = float(np.max(bbox)) / 700.0  
        print(f"Before Downsampling {len(pcd.points)}")
        print(f"[Dense] Voxel downsampling (size={voxel_size:.4f})")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Before statistical removal {len(pcd.points)}")
        # -------------------------------------------------------
        print("[Dense] Statistical outlier removal")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"[Dense] Retained {len(pcd.points)} pts after cleanup")

    # Estimate surface normals so downstream meshing / shading works
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[Dense] Saved {output_path}")

    if show:
        o3d.visualization.draw_geometries([pcd], window_name="Dense reconstruction",
                                          width=1280, height=720)
    return pcd
