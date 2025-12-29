import numpy as np
import cv2


def normalize_points(points):
    points = points.reshape(-1, 2)
    centroid = np.mean(points, axis=0)

    # Shift origin to centroid
    points_shifted = points - centroid

    # Scale so that average distance from origin is sqrt(2)
    mean_dist = np.mean(np.sqrt(points_shifted[:, 0]**2 + points_shifted[:, 1]**2))
    if mean_dist > 0:
        scale = np.sqrt(2) / mean_dist
    else:
        scale = 1.0

    # Construct transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Apply transformation
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    points_normalized = (T @ points_homogeneous.T).T[:, :2]

    return points_normalized, T

def compute_fundamental_matrix(pts1, pts2):
    n = len(pts1)

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build the constraint matrix A
    A = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0
    F = U_f @ np.diag(S_f) @ Vt_f

    # Denormalize F
    F = T2.T @ F @ T1

    F = F / F[2, 2]

    return F

def compute_fundamental_matrix_ransac(src_pts, dst_pts, threshold=1.0, max_iters=2000):
    pts1 = src_pts.reshape(-1, 2)
    pts2 = dst_pts.reshape(-1, 2)
    n = pts1.shape[0]

    best_F = None
    best_inliers = []
    best_num_inliers = 0

    for _ in range(max_iters):
        # Sample 8 random points
        indices = np.random.choice(n, 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Compute F from these 8 points
        F = compute_fundamental_matrix(
            sample_pts1.reshape(-1, 1, 2),
            sample_pts2.reshape(-1, 1, 2)
        )

        # Test on all points
        pts1_h = np.hstack([pts1, np.ones((n, 1))])
        pts2_h = np.hstack([pts2, np.ones((n, 1))])

        # Compute epipolar lines
        epilines2 = (F @ pts1_h.T).T
        epilines1 = (F.T @ pts2_h.T).T

        # Point-to-line distances
        dist1 = np.abs(np.sum(pts1_h * epilines1, axis=1)) / np.sqrt(epilines1[:, 0]**2 + epilines1[:, 1]**2)
        dist2 = np.abs(np.sum(pts2_h * epilines2, axis=1)) / np.sqrt(epilines2[:, 0]**2 + epilines2[:, 1]**2)

        # Inliers are points with small distances in both images
        inliers = (dist1 < threshold) & (dist2 < threshold)
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_F = F

    # Refine F using all inliers
    if best_num_inliers >= 8:
        inlier_pts1 = pts1[best_inliers].reshape(-1, 1, 2)
        inlier_pts2 = pts2[best_inliers].reshape(-1, 1, 2)
        best_F = compute_fundamental_matrix(inlier_pts1, inlier_pts2)

    return best_F, best_inliers

def compute_essential_matrix(F, K, threshold=1.0):
    E = K.T @ F @ K
    # Enforce the essential matrix constraints
    U, S, Vt = np.linalg.svd(E)
    # Essential matrix has two equal singular values and one zero
    S = np.array([1, 1, 0])
    E = U @ np.diag(S) @ Vt

    return E

K = np.array([[1920, 0, 1920/2],
            [0, 1920, 1080/2],
            [0,  0,  1]])

def compute_fundamental_matrix_cv(src_pts, dst_pts, threshold=1.0, max_iters=2000):
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold, confidence=0.999, maxIters=max_iters)

    # Get inlier points
    if mask is not None:
        src_inliers = src_pts[mask.ravel() == 1]
        dst_inliers = dst_pts[mask.ravel() == 1]
    else:
        src_inliers = src_pts
        dst_inliers = dst_pts
        mask = np.ones((len(src_pts), 1))

    return F, mask, src_inliers, dst_inliers

def compute_essential_matrix_cv(src_pts, dst_pts, K, threshold=1.0):
    E, inlier_mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, threshold=threshold, prob=0.999)

    # Get inlier points
    if inlier_mask is not None:
        src_inliers = src_pts[inlier_mask.ravel() == 1]
        dst_inliers = dst_pts[inlier_mask.ravel() == 1]
    else:
        src_inliers = src_pts
        dst_inliers = dst_pts

    return E, inlier_mask, src_inliers, dst_inliers