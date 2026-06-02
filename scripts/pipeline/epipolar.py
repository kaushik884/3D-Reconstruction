# Kaushik Deo
# April 23 2026
"""Two-view geometry: F, E, pose recovery, triangulation, and epipolar-line visualization."""
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_fundamental(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 1.0,
    max_iters: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate F with USAC_MAGSAC and return (F, mask, src_inliers, dst_inliers)."""
    # MAGSAC++ is robust to outliers and avoids the need for a hard inlier threshold
    F, mask = cv2.findFundamentalMat(
        src_pts,
        dst_pts,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999,
        maxIters=max_iters,
    )
    if mask is None:
        mask = np.ones((len(src_pts), 1), dtype=np.uint8)
    inlier_mask = mask.ravel().astype(bool)
    return F, inlier_mask, src_pts[inlier_mask], dst_pts[inlier_mask]


def compute_essential(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate E with RANSAC and return (E, mask, src_inliers, dst_inliers)."""
    E, mask = cv2.findEssentialMat(
        src_pts,
        dst_pts,
        K,
        method=cv2.RANSAC,
        threshold=threshold,
        prob=0.999,
    )
    if mask is None:
        mask = np.ones((len(src_pts), 1), dtype=np.uint8)
    inlier_mask = mask.ravel().astype(bool)
    return E, inlier_mask, src_pts[inlier_mask], dst_pts[inlier_mask]


def recover_pose(
    E: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose E via cheirality. Returns (R, t, pose_mask, src_inliers, dst_inliers)."""
    # Picks the (R, t) solution where triangulated points lie in front of both cameras
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
    pose_mask = mask.ravel() != 0
    return R, t, pose_mask, src_pts[pose_mask], dst_pts[pose_mask]


def triangulate(
    P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
) -> np.ndarray:
    """Linear DLT triangulation via OpenCV. Returns (N, 3)."""
    p1 = pts1.reshape(-1, 2).T
    p2 = pts2.reshape(-1, 2).T
    pts4d = cv2.triangulatePoints(P1, P2, p1, p2)
    # Homogeneous -> Euclidean 3D coordinates
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def reprojection_errors(P: np.ndarray, pts_3d: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """Per-point reprojection error (L2 pixels). Points behind camera return inf."""
    n = len(pts_3d)
    homo = np.hstack([pts_3d, np.ones((n, 1))])
    proj = (P @ homo.T).T
    errs = np.full(n, np.inf)
    valid = proj[:, 2] > 1e-6
    proj_2d = proj[valid, :2] / proj[valid, 2:3]
    errs[valid] = np.linalg.norm(proj_2d - pts_2d.reshape(-1, 2)[valid], axis=1)
    return errs


def draw_epipolar_lines(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    F: np.ndarray,
    img_src: np.ndarray,
    img_dst: np.ndarray,
    max_draw: int = 50,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Draw epipolar lines corresponding to a random subset of point matches."""
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    height, width = img_src.shape[:2]

    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    if len(src) > max_draw:
        idx = np.random.choice(len(src), max_draw, replace=False)
        src = src[idx]
        dst = dst[idx]

    # Compute epipolar lines in image 2 for points in image 1 (and vice versa)
    lines_dst = cv2.computeCorrespondEpilines(src.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines_src = cv2.computeCorrespondEpilines(dst.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    for i in range(len(src)):
        color = tuple(np.random.rand(3))
        _draw_line(ax[1], lines_dst[i], width, height, color)
        _draw_line(ax[0], lines_src[i], width, height, color)

    ax[0].scatter(src[:, 0], src[:, 1], s=10, c="lime", edgecolors="black", linewidth=0.5)
    ax[1].scatter(dst[:, 0], dst[:, 1], s=10, c="lime", edgecolors="black", linewidth=0.5)

    ax[0].imshow(img_src); ax[0].set_title(f"Image 1 - {len(src)} correspondences"); ax[0].axis("off")
    ax[1].imshow(img_dst); ax[1].set_title(f"Image 2 - {len(src)} correspondences"); ax[1].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved epipolar viz to {save_path}")
    if show:
        plt.show()
    plt.close()


def _draw_line(ax, line, width, height, color) -> None:
    a, b, c = line
    if abs(b) > 1e-6:
        x0, y0 = 0, -c / b
        x1, y1 = width, -(c + a * width) / b
    else:
        x0, x1 = -c / a, -c / a
        y0, y1 = 0, height
    ax.plot([x0, x1], [y0, y1], c=color, alpha=0.5, linewidth=1)
