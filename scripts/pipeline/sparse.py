# Kaushik Deo
# April 23 2026
"""Incremental Structure-from-Motion. Tracks correspondences by keypoint index."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from . import epipolar

def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)

def _directional(match: Dict, new_idx: int, existing_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (new_pts, existing_pts, new_kp_idx, existing_kp_idx)"""
    if new_idx < existing_idx:
        return match["src_pts"], match["dst_pts"], match["src_kp_idx"], match["dst_kp_idx"]
    return match["dst_pts"], match["src_pts"], match["dst_kp_idx"], match["src_kp_idx"]


def find_best_initial_pair(
    all_matches: Dict, K: np.ndarray, threshold: float, max_iters: int, min_matches: int = 50
) -> Tuple[int, int]:
    """Pick the pair with the highest score = F_inliers * (1 - H/F ratio)."""
    # Low H/F ratio => scene has real parallax, so triangulation will be stable
    candidates = []
    for (i, j), m in all_matches.items():
        src, dst = m["src_pts"], m["dst_pts"]
        if len(src) < min_matches:
            continue
        F, F_mask, _, _ = epipolar.compute_fundamental(src, dst, threshold, max_iters)
        if F is None:
            continue
        H, H_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H_mask is None:
            continue
        F_inliers = int(F_mask.sum())
        H_inliers = int(H_mask.sum())
        if F_inliers < min_matches:
            continue
        # If a homography fits nearly as well as F, the scene is near-planar (reject)
        ratio = H_inliers / max(F_inliers, 1)
        if ratio >= 0.55:
            continue
        score = F_inliers * (1.0 - ratio)
        candidates.append((score, (i, j), ratio, F_inliers))

    if not candidates:
        raise RuntimeError("No suitable initial pair found (all pairs too degenerate).")
    candidates.sort(reverse=True)
    score, best_pair, ratio, inliers = candidates[0]
    print(f"Best initial pair {best_pair}: score={score:.1f}, H/F ratio={ratio:.2f}, F inliers={inliers}")
    return best_pair


def initialize_reconstruction(
    idx1: int,
    idx2: int,
    all_matches: Dict,
    K: np.ndarray,
    reproj_max: float = 5.0,
    ransac_threshold: float = 1.0,
) -> Dict:
    """Two-view initialization using E + cheirality + reprojection filter."""
    i, j = _pair_key(idx1, idx2)
    m = all_matches[(i, j)]
    src, dst = m["src_pts"], m["dst_pts"]
    ki, kj = m["src_kp_idx"], m["dst_kp_idx"]

    # Essential matrix encodes relative pose for calibrated cameras
    E, e_mask, _, _ = epipolar.compute_essential(src, dst, K, ransac_threshold)
    if E is None:
        raise RuntimeError(f"Essential matrix estimation failed for pair ({i}, {j}).")

    # Recover R, t (relative pose) from E using cheirality check
    _, R, t, pose_mask = cv2.recoverPose(E, src, dst, K)
    # Intersect the essential-matrix inlier mask with the cheirality mask
    pose_mask = pose_mask.ravel() != 0
    final_mask = e_mask & pose_mask

    src_in, dst_in = src[final_mask], dst[final_mask]
    ki_in, kj_in = ki[final_mask], kj[final_mask]

    # First camera anchors the world frame at the origin; second camera uses recovered (R, t)
    P1 = epipolar.projection_matrix(K, np.eye(3), np.zeros((3, 1)))
    P2 = epipolar.projection_matrix(K, R, t)

    # Triangulate and filter points with high reprojection error or negative depth
    pts_3d = epipolar.triangulate(P1, P2, src_in, dst_in)
    err1 = epipolar.reprojection_errors(P1, pts_3d, src_in)
    err2 = epipolar.reprojection_errors(P2, pts_3d, dst_in)
    good = (err1 < reproj_max) & (err2 < reproj_max) & (pts_3d[:, 2] > 0)

    pts_3d = pts_3d[good].tolist()
    src_in, dst_in = src_in[good], dst_in[good]
    ki_in, kj_in = ki_in[good], kj_in[good]

    print(f"Initial pair ({i},{j}): {len(pts_3d)} / {len(src)} points triangulated")

    reconstruction = {
        "cameras": {
            i: {"R": np.eye(3), "t": np.zeros((3, 1)), "P": P1},
            j: {"R": R, "t": t, "P": P2},
        },
        "points_3d": pts_3d,
        # cam_idx -> {kp_idx -> point_3d_idx}
        "obs": {i: {}, j: {}},
        # point_3d_idx -> {cam_idx -> kp_idx}
        "point_views": {},
        # cached 2D observation dict: cam_idx -> {point_3d_idx -> (x, y)}
        "observations": {i: {}, j: {}},
    }
    for pt_idx in range(len(pts_3d)):
        kp_i, kp_j = int(ki_in[pt_idx]), int(kj_in[pt_idx])
        reconstruction["obs"][i][kp_i] = pt_idx
        reconstruction["obs"][j][kp_j] = pt_idx
        reconstruction["point_views"][pt_idx] = {i: kp_i, j: kp_j}
        reconstruction["observations"][i][pt_idx] = src_in[pt_idx].ravel()
        reconstruction["observations"][j][pt_idx] = dst_in[pt_idx].ravel()
    return reconstruction


def _find_2d_3d(reconstruction: Dict, new_idx: int, all_matches: Dict) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Collect 2D-3D correspondences via keypoint-index lookup."""
    pts_2d, pts_3d, pt_indices, kp_indices = [], [], [], []
    seen_pts = set()

    for cam_idx in reconstruction["cameras"]:
        pair = _pair_key(new_idx, cam_idx)
        if pair not in all_matches:
            continue
        m = all_matches[pair]
        _, _, new_kp_idx, exist_kp_idx = _directional(m, new_idx, cam_idx)
        new_pts, _, _, _ = _directional(m, new_idx, cam_idx)

        cam_obs = reconstruction["obs"].get(cam_idx, {})
        for k_new, k_ex, pt_new in zip(new_kp_idx, exist_kp_idx, new_pts):
            pt3_idx = cam_obs.get(int(k_ex))
            if pt3_idx is not None and pt3_idx not in seen_pts:
                seen_pts.add(pt3_idx)
                pts_2d.append(pt_new.ravel())
                pts_3d.append(reconstruction["points_3d"][pt3_idx])
                pt_indices.append(pt3_idx)
                kp_indices.append(int(k_new))

    if len(pts_2d) == 0:
        return np.empty((0, 2)), np.empty((0, 3)), [], []
    return np.array(pts_2d, dtype=np.float32), np.array(pts_3d, dtype=np.float32), pt_indices, kp_indices


def _select_next_camera(reconstruction: Dict, remaining: set, all_matches: Dict, min_corr: int = 10) -> Optional[int]:
    best_cam, best_count = None, -1
    for cam_idx in remaining:
        pts_2d, *_ = _find_2d_3d(reconstruction, cam_idx, all_matches)
        if len(pts_2d) >= min_corr and len(pts_2d) > best_count:
            best_count = len(pts_2d)
            best_cam = cam_idx
    return best_cam


def _add_camera_pnp(
    reconstruction: Dict,
    new_idx: int,
    all_matches: Dict,
    K: np.ndarray,
    pnp_reproj: float = 3.0,
) -> bool:
    pts_2d, pts_3d, pt_indices, kp_indices = _find_2d_3d(reconstruction, new_idx, all_matches)
    if len(pts_2d) < 8:
        return False

    # PnP: solve for the new camera's pose given 2D-3D correspondences from already-built map
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, None,
        iterationsCount=2000, reprojectionError=pnp_reproj, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 15:
        print(f"  Camera {new_idx}: PnP failed ({len(inliers) if inliers is not None else 0} inliers)")
        return False

    # Rodrigues converts axis-angle rotation vector into a 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    P = epipolar.projection_matrix(K, R, t)
    reconstruction["cameras"][new_idx] = {"R": R, "t": t, "P": P}
    reconstruction["obs"].setdefault(new_idx, {})
    reconstruction["observations"].setdefault(new_idx, {})

    for inlier_i in inliers.ravel():
        pt3_idx = pt_indices[inlier_i]
        kp_new = kp_indices[inlier_i]
        reconstruction["obs"][new_idx][kp_new] = pt3_idx
        reconstruction["point_views"][pt3_idx][new_idx] = kp_new
        reconstruction["observations"][new_idx][pt3_idx] = pts_2d[inlier_i]

    C = -R.T @ t
    print(f"  Camera {new_idx}: {len(inliers)} PnP inliers, center=[{C[0,0]:.2f} {C[1,0]:.2f} {C[2,0]:.2f}]")
    return True


def _triangulate_new_points(
    reconstruction: Dict,
    new_idx: int,
    all_matches: Dict,
    keypoints: Sequence,
    reproj_max: float = 3.0,
) -> int:
    new_cam = reconstruction["cameras"][new_idx]
    P_new = new_cam["P"]
    C_new = (-new_cam["R"].T @ new_cam["t"]).ravel()
    added = 0

    for cam_idx in list(reconstruction["cameras"].keys()):
        if cam_idx == new_idx:
            continue
        pair = _pair_key(new_idx, cam_idx)
        if pair not in all_matches:
            continue
        m = all_matches[pair]
        new_pts, exist_pts, new_kp_idx, exist_kp_idx = _directional(m, new_idx, cam_idx)

        exist_obs = reconstruction["obs"].get(cam_idx, {})
        new_obs = reconstruction["obs"].setdefault(new_idx, {})

        to_triangulate_src, to_triangulate_dst = [], []
        kp_new_list, kp_ex_list = [], []
        for k_new, k_ex, p_new, p_ex in zip(new_kp_idx, exist_kp_idx, new_pts, exist_pts):
            k_new, k_ex = int(k_new), int(k_ex)
            if k_new in new_obs:
                continue
            # If the matched keypoint in the existing view is already a 3D point, just extend its track
            if k_ex in exist_obs:
                pt3_idx = exist_obs[k_ex]
                pt_3d = reconstruction["points_3d"][pt3_idx]
                homo = np.append(pt_3d, 1.0)
                proj = P_new @ homo
                
                # Only add if it's in front of the camera AND projects close to the SIFT keypoint
                if proj[2] > 1e-6:
                    proj_2d = proj[:2] / proj[2]
                    err = np.linalg.norm(proj_2d - p_new.ravel())
                    if err < reproj_max:
                        new_obs[k_new] = pt3_idx
                        reconstruction["point_views"][pt3_idx][new_idx] = k_new
                        reconstruction["observations"][new_idx][pt3_idx] = p_new.ravel()
                continue
            to_triangulate_src.append(p_ex.ravel())
            to_triangulate_dst.append(p_new.ravel())
            kp_new_list.append(k_new)
            kp_ex_list.append(k_ex)

        if not to_triangulate_src:
            continue

        P_exist = reconstruction["cameras"][cam_idx]["P"]
        src = np.array(to_triangulate_src, dtype=np.float32)
        dst = np.array(to_triangulate_dst, dtype=np.float32)
        # Triangulate brand-new 3D points from the pair's unmatched keypoints
        pts_3d = epipolar.triangulate(P_exist, P_new, src, dst)

        C_ex = (-reconstruction["cameras"][cam_idx]["R"].T @ reconstruction["cameras"][cam_idx]["t"]).ravel()
        baseline = np.linalg.norm(C_new - C_ex)

        err_exist = epipolar.reprojection_errors(P_exist, pts_3d, src)
        err_new = epipolar.reprojection_errors(P_new, pts_3d, dst)

        dist_exist = np.linalg.norm(pts_3d - C_ex, axis=1)
        dist_new = np.linalg.norm(pts_3d - C_new, axis=1)

        # Reject points with poor reprojection or absurd depth relative to the baseline
        good = (
            (err_exist < reproj_max) & (err_new < reproj_max)
            & (dist_exist > 0.1 * baseline) & (dist_exist < 50 * baseline)
            & (dist_new > 0.1 * baseline) & (dist_new < 50 * baseline)
        )
        good_idx = np.where(good)[0]

        for gi in good_idx:
            pt3_idx = len(reconstruction["points_3d"])
            reconstruction["points_3d"].append(pts_3d[gi])
            reconstruction["obs"][cam_idx][kp_ex_list[gi]] = pt3_idx
            reconstruction["obs"][new_idx][kp_new_list[gi]] = pt3_idx
            reconstruction["point_views"][pt3_idx] = {cam_idx: kp_ex_list[gi], new_idx: kp_new_list[gi]}
            reconstruction["observations"].setdefault(cam_idx, {})[pt3_idx] = to_triangulate_src[gi]
            reconstruction["observations"].setdefault(new_idx, {})[pt3_idx] = to_triangulate_dst[gi]
            added += 1

    print(f"  Triangulated {added} new points")
    return added


def run_sparse_sfm(
    keypoints: Sequence,
    all_matches: Dict,
    K: np.ndarray,
    cfg=None,
) -> Dict:
    """Full incremental SfM. `cfg` is the pipeline Config (provides thresholds)."""
    ransac_thresh = getattr(cfg, "ransac_threshold_px", 1.0) if cfg else 1.0
    ransac_iters = getattr(cfg, "ransac_max_iters", 2000) if cfg else 2000
    reproj_max_init = getattr(cfg, "triangulation_reproj_max", 5.0) if cfg else 5.0
    pnp_reproj = getattr(cfg, "pnp_reprojection_error", 5.0) if cfg else 5.0

    num_images = len(keypoints)

    # Step 1: pick the pair with strongest parallax + enough matches
    print("\n[Sparse] Step 1: Finding best initial pair")
    best_pair = find_best_initial_pair(all_matches, K, ransac_thresh, ransac_iters)

    # Step 2: two-view seed reconstruction (triangulate initial point cloud)
    print(f"[Sparse] Step 2: Initializing with pair {best_pair}")
    reconstruction = initialize_reconstruction(
        best_pair[0], best_pair[1], all_matches, K, reproj_max_init, ransac_thresh
    )

    # Step 3: incrementally register each remaining view via PnP and triangulate new points
    print("[Sparse] Step 3: Adding remaining cameras")
    processed = set(best_pair)
    remaining = set(range(num_images)) - processed

    while remaining:
        # Greedy: pick the unregistered camera with the most 2D-3D overlap with current map
        next_cam = _select_next_camera(reconstruction, remaining, all_matches)
        if next_cam is None:
            print(f"  Stopping early: {len(remaining)} cameras could not be registered.")
            break
        ok = _add_camera_pnp(reconstruction, next_cam, all_matches, K, pnp_reproj)
        remaining.remove(next_cam)
        if ok:
            processed.add(next_cam)
            _triangulate_new_points(reconstruction, next_cam, all_matches, keypoints, reproj_max=3.0)
    # Reject cameras whose optical centers are far from the cluster (likely mis-registered)
    centers = np.array([
        (-cam["R"].T @ cam["t"]).ravel()
        for cam in reconstruction["cameras"].values()
    ])
    median_center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median_center, axis=1)
    mad = np.median(distances)  # median absolute deviation

    bad_cams = []
    cam_indices = sorted(reconstruction["cameras"].keys())
    for idx, cam_idx in enumerate(cam_indices):
        # MAD-based outlier test is more robust than standard deviation
        if distances[idx] > 5.0 * mad:
            bad_cams.append(cam_idx)
            print(f"[Sparse] Removing outlier camera {cam_idx} "
                f"(distance={distances[idx]:.2f}, MAD={mad:.2f})")

    for cam_idx in bad_cams:
        del reconstruction["cameras"][cam_idx]
        reconstruction["obs"].pop(cam_idx, None)
        reconstruction["observations"].pop(cam_idx, None)

    print(f"[Sparse] Removed {len(bad_cams)} outlier cameras")
    reconstruction["points_3d"] = np.array(reconstruction["points_3d"])
    print(f"[Sparse] Final: {len(reconstruction['cameras'])} cameras, "
          f"{len(reconstruction['points_3d'])} points")
    return reconstruction
