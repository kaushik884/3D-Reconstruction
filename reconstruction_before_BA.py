import numpy as np
import cv2
import F_E_matrices
import sift
import triangulate
import read_images as ri
import visualize_3D

def compute_matches_matrix(keypoints, descriptors):
    num_images = len(keypoints)
    matches_matrix = np.zeros((num_images, num_images), dtype=int)
    all_matches = {}

    for i in range(num_images):
        for j in range(i+1, num_images):
            src_pts, dst_pts, good_matches = sift.match_features(i, j, keypoints, descriptors)

            if len(good_matches) > 0:
                matches_matrix[i, j] = len(good_matches)
                matches_matrix[j, i] = len(good_matches)
                all_matches[(i, j)] = (src_pts, dst_pts, good_matches)

    return matches_matrix, all_matches

def find_best_initial_pair(matches_matrix, all_matches, K):
    num_images = matches_matrix.shape[0]
    candidates = []

    # Check all pairs
    for i in range(num_images):
        for j in range(i+1, num_images):

            src_pts, dst_pts, _ = all_matches[(i, j)]

            # Find fundamental matrix
            F, F_mask = F_E_matrices.compute_fundamental_matrix_ransac(src_pts, dst_pts, 1.0, 2000)

            F_mask = F_mask.astype(np.uint8).reshape(-1, 1)
            F_src_inliers = src_pts[F_mask.ravel() == 1]
            F_dst_inliers = dst_pts[F_mask.ravel() == 1]

            # Find homography
            H, H_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

            F_inliers = np.sum(F_mask)
            H_inliers = np.sum(H_mask)

            # CRITICAL: Check homography ratio - low ratio means good baseline
            ratio = H_inliers / F_inliers

            # Good baseline: ratio should be < 0.7
            if ratio < 0.7:
                E = F_E_matrices.compute_essential_matrix(F, K, 1.0)
                if E is not None:
                    F_inliers = len(F_src_inliers)
                    if F_inliers > 30:
                        score = F_inliers * (1 - ratio)
                        candidates.append((score, (i, j), ratio))

    candidates.sort(reverse=True)
    best_score, best_pair, best_ratio = candidates[0]
    print(f"Selected pair {best_pair} with score {best_score:.1f}, H/F ratio {best_ratio:.2f}")

    return best_pair

def initialize_reconstruction(idx1, idx2, all_matches, K):
    src_pts, dst_pts, good_matches = all_matches[(min(idx1, idx2), max(idx1, idx2))]

    # Find essential matrix
    F, F_mask = F_E_matrices.compute_fundamental_matrix_ransac(src_pts, dst_pts, 1.0, 2000)
    F_mask = F_mask.astype(np.uint8).reshape(-1, 1)
    src_inliers = src_pts[F_mask.ravel() == 1]
    dst_inliers = dst_pts[F_mask.ravel() == 1]

    E = F_E_matrices.compute_essential_matrix(F, K, 1.0)

    # Recover pose
    (R, t), initial_pts_3d = triangulate.find_correct_pose(E, K, src_inliers, dst_inliers)

    # CRITICAL: Normalize translation to unit length
    t = t / np.linalg.norm(t)

    print(f"Initial pair: {len(src_inliers)} inliers from {len(src_pts)} matches")

    # Setup projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Triangulate with validation
    pts_3d = []
    valid_src = []
    valid_dst = []

    pts_3d_all = triangulate.triangulate_points(P1, P2, src_inliers, dst_inliers)

    for i, pt_3d in enumerate(pts_3d_all):
        # Check reprojection error
        pt_homo = np.append(pt_3d, 1)
        pt_proj1 = P1 @ pt_homo
        pt_proj1 = pt_proj1[:2] / pt_proj1[2]
        err1 = np.linalg.norm(pt_proj1 - src_inliers[i].ravel())

        pt_proj2 = P2 @ pt_homo
        pt_proj2 = pt_proj2[:2] / pt_proj2[2]
        err2 = np.linalg.norm(pt_proj2 - dst_inliers[i].ravel())

        if err1 < 5.0 and err2 < 5.0:  # Pixel threshold
            pts_3d.append(pt_3d)
            valid_src.append(src_inliers[i].ravel())
            valid_dst.append(dst_inliers[i].ravel())

    pts_3d = np.array(pts_3d)
    print(f"Triangulated {len(pts_3d)} valid points")

    # Build reconstruction structure
    reconstruction = {
        'cameras': {
            idx1: {'R': np.eye(3), 't': np.zeros((3, 1)), 'P': P1},
            idx2: {'R': R, 't': t, 'P': P2}
        },
        'points_3d': pts_3d,
        'observations': {
            idx1: {i: pt for i, pt in enumerate(valid_src)},
            idx2: {i: pt for i, pt in enumerate(valid_dst)}
        },
        'point_to_observations': {
            i: {idx1: src_pt, idx2: dst_pt}
            for i, (src_pt, dst_pt) in enumerate(zip(valid_src, valid_dst))
        }
    }

    return reconstruction

def find_2d_3d_correspondences(reconstruction, new_img_idx, all_matches):
    points_2d = []
    points_3d = []
    point_indices = []

    for existing_cam_idx in reconstruction['cameras']:
        if existing_cam_idx >= new_img_idx:
            match_key = (new_img_idx, existing_cam_idx)
            src_idx = 0
        else:
            match_key = (existing_cam_idx, new_img_idx)
            src_idx = 1

        if match_key not in all_matches:
            continue

        src_pts, dst_pts, matches = all_matches[match_key]

        if src_idx == 0:
            new_img_pts = src_pts
            exist_img_pts = dst_pts
        else:
            new_img_pts = dst_pts
            exist_img_pts = src_pts

        # Match with existing observations
        for pt_3d_idx, stored_2d in reconstruction['observations'][existing_cam_idx].items():
            if pt_3d_idx >= len(reconstruction['points_3d']):
                continue

            # Find closest match
            distances = np.linalg.norm(exist_img_pts.reshape(-1, 2) - stored_2d, axis=1)
            min_idx = np.argmin(distances)

            if distances[min_idx] < 3.0 and pt_3d_idx not in point_indices:
                points_2d.append(new_img_pts[min_idx].ravel())
                points_3d.append(reconstruction['points_3d'][pt_3d_idx])
                point_indices.append(pt_3d_idx)

    return np.array(points_2d), np.array(points_3d), point_indices

def add_camera_pnp(reconstruction, new_img_idx, all_matches, K):
    points_2d, points_3d, point_indices = find_2d_3d_correspondences(reconstruction, new_img_idx, all_matches)

    # Use RANSAC PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float32),
        points_2d.astype(np.float32),
        K.astype(np.float32),
        None,
        iterationsCount=1000,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < 8:
        print(f"Camera {new_img_idx}: PnP failed")
        return False

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    camera_center = -R.T @ t

    # Add camera
    P = K @ np.hstack([R, t])
    reconstruction['cameras'][new_img_idx] = {'R': R, 't': t, 'P': P}

    if new_img_idx not in reconstruction['observations']:
        reconstruction['observations'][new_img_idx] = {}

    # Store observations for inlier points
    for idx in inliers.ravel():
        pt_3d_idx = point_indices[idx]
        reconstruction['observations'][new_img_idx][pt_3d_idx] = points_2d[idx]
        if pt_3d_idx not in reconstruction['point_to_observations']:
            reconstruction['point_to_observations'][pt_3d_idx] = {}
        reconstruction['point_to_observations'][pt_3d_idx][new_img_idx] = points_2d[idx]

    print(f"Added camera {new_img_idx}: {len(inliers)} inliers, center at {camera_center.T}")

    # Triangulate new points with existing cameras
    triangulate_new_points(reconstruction, new_img_idx, all_matches, K)

    return True

def triangulate_new_points(reconstruction, new_cam_idx, all_matches, K):
    new_cam = reconstruction['cameras'][new_cam_idx]
    num_new_points = 0

    # Convert to list if needed
    if isinstance(reconstruction['points_3d'], np.ndarray):
        reconstruction['points_3d'] = reconstruction['points_3d'].tolist()

    for existing_cam_idx in reconstruction['cameras']:
        if existing_cam_idx == new_cam_idx:
            continue

        # Get matches
        if existing_cam_idx < new_cam_idx:
            match_key = (existing_cam_idx, new_cam_idx)
            if match_key not in all_matches:
                continue
            src_pts, dst_pts, _ = all_matches[match_key]
            exist_pts = src_pts
            new_pts = dst_pts
        else:
            match_key = (new_cam_idx, existing_cam_idx)
            if match_key not in all_matches:
                continue
            src_pts, dst_pts, _ = all_matches[match_key]
            exist_pts = dst_pts
            new_pts = src_pts

        P1 = reconstruction['cameras'][existing_cam_idx]['P']
        P2 = new_cam['P']

        # Collect points to triangulate
        points_to_triangulate_1 = []
        points_to_triangulate_2 = []
        indices = []

        for i, (new_pt, exist_pt) in enumerate(zip(new_pts, exist_pts)):
            # Check if already observed
            already_exists = False
            for obs_2d in reconstruction['observations'].get(existing_cam_idx, {}).values():
                if np.linalg.norm(obs_2d - exist_pt.ravel()) < 2.0:
                    already_exists = True
                    break

            if not already_exists:
                points_to_triangulate_1.append(exist_pt.ravel())
                points_to_triangulate_2.append(new_pt.ravel())
                indices.append(i)

        # Triangulate new points
        if len(points_to_triangulate_1) > 0:
            pts_3d = triangulate.triangulate_points(P1, P2, np.array(points_to_triangulate_1), np.array(points_to_triangulate_2))

            # Validate each point
            cam1_center = -reconstruction['cameras'][existing_cam_idx]['R'].T @ \
                         reconstruction['cameras'][existing_cam_idx]['t']
            cam2_center = -new_cam['R'].T @ new_cam['t']
            baseline = np.linalg.norm(cam1_center - cam2_center)

            for i, pt_3d in enumerate(pts_3d):
                # Check depth
                dist1 = np.linalg.norm(pt_3d - cam1_center.ravel())
                dist2 = np.linalg.norm(pt_3d - cam2_center.ravel())

                # Points should be at reasonable distance
                if dist1 < 0.1 * baseline or dist1 > 50 * baseline:
                    continue
                if dist2 < 0.1 * baseline or dist2 > 50 * baseline:
                    continue

                # Check reprojection error
                pt_homo = np.append(pt_3d, 1)
                reproj1 = P1 @ pt_homo
                reproj1 = reproj1[:2] / reproj1[2]
                err1 = np.linalg.norm(reproj1 - points_to_triangulate_1[i])

                reproj2 = P2 @ pt_homo
                reproj2 = reproj2[:2] / reproj2[2]
                err2 = np.linalg.norm(reproj2 - points_to_triangulate_2[i])

                if err1 < 3.0 and err2 < 3.0:
                    # Add point
                    pt_idx = len(reconstruction['points_3d'])
                    reconstruction['points_3d'].append(pt_3d)

                    if existing_cam_idx not in reconstruction['observations']:
                        reconstruction['observations'][existing_cam_idx] = {}
                    if new_cam_idx not in reconstruction['observations']:
                        reconstruction['observations'][new_cam_idx] = {}

                    reconstruction['observations'][existing_cam_idx][pt_idx] = points_to_triangulate_1[i]
                    reconstruction['observations'][new_cam_idx][pt_idx] = points_to_triangulate_2[i]

                    num_new_points += 1

    print(f"Triangulated {num_new_points} new points")

def select_next_camera(reconstruction, remaining, all_matches):
    best_score = -1
    best_cam = None

    for cam_idx in remaining:
        points_2d, points_3d, _ = find_2d_3d_correspondences(reconstruction, cam_idx, all_matches)

        if len(points_2d) >= 15:  # Need good visibility
            score = len(points_2d)
            if score > best_score:
                best_score = score
                best_cam = cam_idx

    return best_cam

def incremental_sfm(keypoints, descriptors, K, gray_images):
    num_images = len(keypoints)

    print("Step 1: Computing pairwise matches")
    matches_matrix, all_matches = compute_matches_matrix(keypoints, descriptors)

    print("\nStep 2: Finding best initial pair")
    best_pair = find_best_initial_pair(matches_matrix, all_matches, K)

    print(f"\nStep 3: Initializing with pair {best_pair}")
    reconstruction = initialize_reconstruction(best_pair[0], best_pair[1], all_matches, K)

    print("\nStep 4: Adding remaining cameras...")
    processed = set(best_pair)
    remaining = set(range(num_images)) - processed

    while remaining:
        next_cam = select_next_camera(reconstruction, remaining, all_matches)

        success = add_camera_pnp(reconstruction, next_cam, all_matches, K)

        remaining.remove(next_cam)
        if success:
            processed.add(next_cam)

    # Convert back to array
    reconstruction['points_3d'] = np.array(reconstruction['points_3d'])

    print(f"\nFinal: {len(reconstruction['cameras'])} cameras, {len(reconstruction['points_3d'])} points")

    return reconstruction

# Run the pipeline
# print("Running SFM pipeline")
# og_images, gray_images, num_images = ri.read_imgs_from_folder("buddha", show_images=False)
# keypoints_, descriptors_ = sift.detect_keypoints_sift(gray_images, num_images, show_images=False)
# keypoints_sf, descriptors_sf = sift.apply_nms(keypoints_, descriptors_, og_images)
# reconstruction = incremental_sfm(keypoints_sf, descriptors_sf, F_E_matrices.K, gray_images)

# # Visualize
# fig = visualize_3D.visualize_reconstruction(reconstruction)
# fig.show()