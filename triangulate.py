import numpy as np
import cv2

def decompose_essential_matrix(E):
    U, S, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrices (det = +1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    # Enforce essential matrix constraint (singular values: 1, 1, 0)
    E_normalized = U @ np.diag([1, 1, 0]) @ Vt

    # W matrix for rotation extraction
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Translation is the last column of U (up to scale)
    t = U[:, 2:3]  # Keep as column vector

    # Four possible solutions (R1/R2 with ±t)
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return solutions

def triangulate_points(P1, P2, pts1, pts2):
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    pts_3d = np.zeros((len(pts1), 4))

    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Build matrix A for DLT
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Normalize by w
        pts_3d[i] = X

    return pts_3d[:, :3]  # Return only X,Y,Z

def find_correct_pose(E, K, pts1, pts2):
    solutions = decompose_essential_matrix(E)
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_solution = None
    best_count = 0
    best_pts_3d = None
    best_R = None
    best_t = None

    for i, (R, t) in enumerate(solutions):
        P2 = K @ np.hstack([R, t])

        # Triangulate points
        pts_3d = triangulate_points(P1, P2, pts1, pts2)

        # Check depths in camera 1 (Z coordinate)
        z1 = pts_3d[:, 2]

        # Transform to camera 2 and check depths
        pts_3d_cam2 = (R @ pts_3d.T + t).T
        z2 = pts_3d_cam2[:, 2]

        # Count points with positive depth in both cameras
        valid = np.sum((z1 > 0) & (z2 > 0))

        print(f"Solution {i}: R det={np.linalg.det(R):.3f}, "
              f"valid points: {valid}/{len(pts1)} "
              f"({100*valid/len(pts1):.1f}%)")

        if valid > best_count:
            best_count = valid
            best_pts_3d = pts_3d
            best_R = R
            best_t = t

    return (best_R, best_t), best_pts_3d

def triangulate_points_cv(P1, P2, pts1, pts2):
    pts1 = pts1.reshape(-1, 2).T
    pts2 = pts2.reshape(-1, 2).T
    pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d

def find_correct_pose_cv(E, src_points, dst_points, K):
    _, R, t, inlier_mask = cv2.recoverPose(E, src_points, dst_points, K)

    P = K @ np.hstack((R, t))
    C = np.hstack((R, t))
    T = np.vstack((C, [0,0,0,1]))

    src_inliers = src_points[inlier_mask.ravel() != 0]
    dst_inliers = dst_points[inlier_mask.ravel() != 0]

    return P, C, T, src_inliers, dst_inliers, R, t