# Kaushik Deo
# April 23 2026
"""Bundle adjustment using GTSAM. Fixes only the first camera; anchors scale by a soft prior on the second camera's
translation direction."""
from __future__ import annotations
from typing import Dict, List
import gtsam
import numpy as np
from gtsam import symbol_shorthand

X = symbol_shorthand.X  # Pose3
L = symbol_shorthand.L  # Point3


def _opencv_to_gtsam(R: np.ndarray, t: np.ndarray) -> gtsam.Pose3:
    # OpenCV stores world->camera (R, t); GTSAM expects camera->world pose (R^T, C)
    C = -R.T @ t  # camera center in world
    return gtsam.Pose3(gtsam.Rot3(R.T), gtsam.Point3(C.ravel()))


def _gtsam_to_opencv(pose: gtsam.Pose3):
    # Invert back from GTSAM camera-to-world pose to OpenCV world-to-camera (R, t)
    R_inv = pose.rotation().matrix()
    C = pose.translation()
    R = R_inv.T
    t = -R @ C.reshape(3, 1)
    return R, t


def _filter_outliers(reconstruction: Dict, max_reproj: float = 5.0) -> List[int]:
    """Keep points whose median per-camera reprojection error is below `max_reproj`."""
    points_3d = np.array(reconstruction["points_3d"])
    valid = []
    for pt_idx, point_3d in enumerate(points_3d):
        errs = []
        for cam_idx, cam in reconstruction["cameras"].items():
            obs = reconstruction["observations"].get(cam_idx, {}).get(pt_idx)
            if obs is None:
                continue
            homo = np.append(point_3d, 1.0)
            proj = cam["P"] @ homo
            if proj[2] <= 0:
                continue
            errs.append(np.linalg.norm(proj[:2] / proj[2] - obs))
        if errs and np.median(errs) < max_reproj:
            valid.append(pt_idx)
    print(f"[BA] Kept {len(valid)}/{len(points_3d)} points after outlier filter")
    return valid


def run_bundle_adjustment(reconstruction: Dict, K: np.ndarray) -> Dict:
    valid_points = _filter_outliers(reconstruction)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pixel_sigma = 2.0
    base_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
    # Huber robust kernel down-weights outlier reprojection residuals
    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise
    )
    # Only fix the first camera. Second camera scale is locked by BA via baseline.
    first_cam_prior = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))

    calibration = gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])

    cam_indices = sorted(reconstruction["cameras"].keys())
    for idx, cam_idx in enumerate(cam_indices):
        cam = reconstruction["cameras"][cam_idx]
        pose = _opencv_to_gtsam(cam["R"], cam["t"])
        initial.insert(X(cam_idx), pose)
        # Anchor the world frame: first camera is near-rigidly fixed at its initial pose
        if idx == 0:
            graph.add(gtsam.PriorFactorPose3(X(cam_idx), pose, first_cam_prior))
        elif idx == 1:
            # Soft translation-direction prior on 2nd camera (prevents scale drift)
            scale_prior = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
            )
            graph.add(gtsam.PriorFactorPose3(X(cam_idx), pose, scale_prior))

    points_3d = np.array(reconstruction["points_3d"])
    num_factors = 0
    # For each surviving 3D point, add a projection factor per observing camera
    for pt_idx in valid_points:
        initial.insert(L(pt_idx), gtsam.Point3(points_3d[pt_idx]))
        for cam_idx in cam_indices:
            obs = reconstruction["observations"].get(cam_idx, {}).get(pt_idx)
            if obs is None:
                continue
            cam_P = reconstruction["cameras"][cam_idx]["P"]
            homo = np.append(points_3d[pt_idx], 1.0)
            proj = cam_P @ homo

            # If the point is behind the camera or reprojection error is massive, skip it
            if proj[2] > 0:
                err = np.linalg.norm((proj[:2]/proj[2]) - obs)
                if err > 10.0:
                    continue
            # Projection factor ties observed pixel to (camera pose, 3D landmark)
            graph.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(float(obs[0]), float(obs[1])),
                    measurement_noise,
                    X(cam_idx),
                    L(pt_idx),
                    calibration,
                )
            )
            num_factors += 1

    print(f"[BA] {len(cam_indices)} cameras, {len(valid_points)} points, {num_factors} factors")

    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    params.setMaxIterations(100)
    params.setVerbosityLM("SUMMARY")

    # Levenberg-Marquardt jointly optimizes all poses + landmarks
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    initial_error = graph.error(initial)
    print(f"[BA] Initial error: {initial_error:.3f}")
    result = optimizer.optimize()
    final_error = graph.error(result)
    print(f"[BA] Final error: {final_error:.3f} "
          f"(reduction {100 * (initial_error - final_error) / max(initial_error, 1e-9):.1f}%)")

    optimized = {
        "cameras": {},
        "points_3d": points_3d.copy(),
        "obs": reconstruction["obs"],
        "point_views": reconstruction["point_views"],
        "observations": reconstruction["observations"],
    }
    for cam_idx in cam_indices:
        R, t = _gtsam_to_opencv(result.atPose3(X(cam_idx)))
        optimized["cameras"][cam_idx] = {"R": R, "t": t, "P": K @ np.hstack([R, t])}
    for pt_idx in valid_points:
        optimized["points_3d"][pt_idx] = np.array(result.atPoint3(L(pt_idx)))
    return optimized
