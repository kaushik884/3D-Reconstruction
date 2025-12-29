import gtsam
import numpy as np
from gtsam import symbol_shorthand
X = symbol_shorthand.X  # Pose3 (camera poses)
L = symbol_shorthand.L  # Point3 (3D points)
import read_images as ri
import sift
from reconstruction_before_BA import incremental_sfm
import F_E_matrices
import visualize_3D

def create_camera_calibration(K):
    """Convert opencv intrinsic matrix to GTSAM Cal3_S2"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

def opencv_to_gtsam_pose(R, t):
    """Convert OpenCV camera pose to GTSAM Pose3
    """
    # OpenCV uses different convention than GTSAM
    # OpenCV: [R|t] transforms world to camera
    # GTSAM: Pose3 represents camera pose in world

    # Camera center in world coordinates
    C = -R.T @ t

    # GTSAM pose is camera-to-world (thanks Abhinav for mentioning this in class), so we need the inverse
    R_gtsam = R.T  # Inverse rotation
    t_gtsam = C.ravel()

    gtsam_rot = gtsam.Rot3(R_gtsam)
    gtsam_trans = gtsam.Point3(t_gtsam)

    return gtsam.Pose3(gtsam_rot, gtsam_trans)

def gtsam_to_opencv_pose(pose):
    """Convert GTSAM Pose3 back to OpenCV format"""
    R_gtsam = pose.rotation().matrix()
    C = pose.translation()

    # Convert back to OpenCV convention
    R = R_gtsam.T
    t = -R @ C.reshape(3, 1)

    return R, t

def filter_outlier_points(reconstruction, max_depth=None, max_reproj_error=5.0):
    """Pre-filter points before BA to remove outliers"""
    points_3d = reconstruction['points_3d']
    if isinstance(points_3d, list):
        points_3d = np.array(points_3d)

    valid_indices = []

    # Get camera centers for depth checking
    camera_centers = []
    for cam_data in reconstruction['cameras'].values():
        C = -cam_data['R'].T @ cam_data['t']
        camera_centers.append(C.ravel())
    camera_centers = np.array(camera_centers)

    # If no max_depth specified, use adaptive threshold
    if max_depth is None:
        # Use 95th percentile of camera distances as reference
        inter_camera_dists = []
        for i in range(len(camera_centers)):
            for j in range(i+1, len(camera_centers)):
                dist = np.linalg.norm(camera_centers[i] - camera_centers[j])
                inter_camera_dists.append(dist)
        if inter_camera_dists:
            baseline_scale = np.percentile(inter_camera_dists, 95)
            max_depth = baseline_scale * 50  # Points shouldn't be more than 50x baseline away

    for point_idx in range(len(points_3d)):
        point_3d = points_3d[point_idx]

        # Check if point is at reasonable distance from cameras
        distances = np.linalg.norm(camera_centers - point_3d, axis=1)
        min_dist = np.min(distances)

        if max_depth and min_dist > max_depth:
            continue

        # Check reprojection error
        errors = []
        for cam_idx, cam_data in reconstruction['cameras'].items():
            if cam_idx in reconstruction['observations']:
                if point_idx in reconstruction['observations'][cam_idx]:
                    P = cam_data['P']
                    point_2d = reconstruction['observations'][cam_idx][point_idx]

                    # Project 3D point
                    point_3d_homo = np.append(point_3d, 1)
                    proj = P @ point_3d_homo
                    if proj[2] > 0:
                        proj_2d = proj[:2] / proj[2]
                        error = np.linalg.norm(proj_2d - point_2d)
                        errors.append(error)

        if errors and np.median(errors) < max_reproj_error:
            valid_indices.append(point_idx)

    print(f"Filtered {len(points_3d) - len(valid_indices)} outlier points, keeping {len(valid_indices)}")
    return valid_indices

def bundle_adjustment_gtsam(reconstruction, K):
    # Filter outliers first
    valid_point_indices = filter_outlier_points(reconstruction)

    # Create factor graph and initial estimates
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Setup noise models
    pixel_sigma = 2.0
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma)
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]))
    point_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 10.0)

    # Add camera poses
    camera_indices = sorted(reconstruction['cameras'].keys())

    for cam_idx in camera_indices:
        cam_data = reconstruction['cameras'][cam_idx]
        R = cam_data['R']
        t = cam_data['t']

        # Convert to GTSAM pose
        pose = opencv_to_gtsam_pose(R, t)

        # Add initial estimate
        initial.insert(X(cam_idx), pose)

        # Fix first camera
        if cam_idx == camera_indices[0]:
            graph.add(gtsam.PriorFactorPose3(X(cam_idx), pose, pose_prior_noise))
            print(f"Fixed camera {cam_idx} as origin")
        elif cam_idx == camera_indices[min(1, len(camera_indices)-1)]:
            # Fix second camera to preserve scale
            graph.add(gtsam.PriorFactorPose3(X(cam_idx), pose, pose_prior_noise))
            print(f"Fixed camera {cam_idx} to preserve scale")

    # Add 3D points with regularization
    points_3d = reconstruction['points_3d']
    if isinstance(points_3d, list):
        points_3d = np.array(points_3d)

    num_factors = 0

    for point_idx in valid_point_indices:
        point_3d = points_3d[point_idx]

        # Add initial estimate
        initial.insert(L(point_idx), gtsam.Point3(point_3d))

        graph.add(gtsam.PriorFactorPoint3(L(point_idx), gtsam.Point3(point_3d), point_prior_noise))

        # Add projection factors
        for cam_idx in camera_indices:
            if cam_idx in reconstruction['observations']:
                if point_idx in reconstruction['observations'][cam_idx]:
                    point_2d = reconstruction['observations'][cam_idx][point_idx]

                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        gtsam.Point2(point_2d[0], point_2d[1]),
                        measurement_noise,
                        X(cam_idx),
                        L(point_idx),
                        gtsam.Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])
                    )

                    graph.add(factor)
                    num_factors += 1

    print(f"Added {len(camera_indices)} cameras and {len(valid_point_indices)} 3D points")
    print(f"Added {num_factors} projection factors")

    # Optimize with careful parameters
    print("\nOptimizing")

    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    params.setMaxIterations(100)
    params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    # Check initial error
    initial_error = graph.error(initial)
    print(f"Initial error: {initial_error:.3f}")

    for i in range(10):  # Do 10 iterations at a time
        result = optimizer.optimize()
        current_error = graph.error(result)
        print(f"After {(i+1)*10} iterations: error = {current_error:.3f}")

        # Check if reconstruction is degrading
        if i > 0:
            # Check camera movement
            max_cam_movement = 0
            for cam_idx in camera_indices[:5]:  # Check first 5 cameras
                old_pose = initial.atPose3(X(cam_idx))
                new_pose = result.atPose3(X(cam_idx))
                movement = np.linalg.norm(old_pose.translation() - new_pose.translation())
                max_cam_movement = max(max_cam_movement, movement)

            if max_cam_movement > 100:  # Cameras moving too much
                print(f"WARNING: Cameras moving too much ({max_cam_movement:.1f}), stopping early")
                result = initial  # Revert to initial
                break

    final_error = graph.error(result)
    print(f"\nOptimization complete!")
    print(f"Final error: {final_error:.3f}")
    print(f"Error reduction: {(initial_error - final_error) / initial_error * 100:.1f}%")

    # 4. Extract optimized values
    optimized_reconstruction = {
        'cameras': {},
        'points_3d': reconstruction['points_3d'].copy(),
        'observations': reconstruction['observations'].copy(),
        'point_to_observations': reconstruction['point_to_observations'].copy()
    }

    # Extract camera poses
    for cam_idx in camera_indices:
        optimized_pose = result.atPose3(X(cam_idx))
        R, t = gtsam_to_opencv_pose(optimized_pose)
        P = K @ np.hstack([R, t])

        optimized_reconstruction['cameras'][cam_idx] = {
            'R': R,
            't': t,
            'P': P
        }

    # Extract 3D points
    optimized_points = points_3d.copy()
    for point_idx in valid_point_indices:
        optimized_point = result.atPoint3(L(point_idx))
        optimized_points[point_idx] = np.array([optimized_point[0],
                                               optimized_point[1],
                                               optimized_point[2]])

    optimized_reconstruction['points_3d'] = optimized_points

    return optimized_reconstruction

print("Running SFM pipeline")
og_images, gray_images, num_images = ri.read_imgs_from_folder("buddha", show_images=False)
keypoints_, descriptors_ = sift.detect_keypoints_sift(gray_images, num_images, show_images=False)
keypoints_sf, descriptors_sf = sift.apply_nms(keypoints_, descriptors_, og_images)
reconstruction = incremental_sfm(keypoints_sf, descriptors_sf, F_E_matrices.K, gray_images)
print("Running Bundle Adjustment with GTSAM")
optimized_reconstruction = bundle_adjustment_gtsam(reconstruction, F_E_matrices.K)

fig = visualize_3D.visualize_reconstruction(optimized_reconstruction)
fig.show()