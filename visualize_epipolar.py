import numpy as np
import cv2
import matplotlib.pyplot as plt
import F_E_matrices
import sift 
import read_images as ri

def draw_epipolar_lines(src_inliers, dst_inliers, F, img_src, img_dst):
	_, ax = plt.subplots(1, 2, figsize=(20, 10))

	height, width, _ = img_src.shape

	# Ensure points are in correct format
	src_inliers = src_inliers.reshape(-1, 2)
	dst_inliers = dst_inliers.reshape(-1, 2)

	# Compute epipolar lines
	lines_dst = cv2.computeCorrespondEpilines(src_inliers.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
	lines_src = cv2.computeCorrespondEpilines(dst_inliers.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

	for idx in range(len(src_inliers)):
			color = tuple(np.random.rand(3))

			# Draw line in dst image (right)
			a, b, c = lines_dst[idx]
			# Handle vertical lines
			if abs(b) > 1e-6:
					x_0, y_0 = 0, -c/b
					x_1, y_1 = width, -(c + a*width)/b
			else:
					# Vertical line
					x_0, x_1 = -c/a, -c/a
					y_0, y_1 = 0, height

			ax[1].plot([x_0, x_1], [y_0, y_1], c=color, alpha=0.5, linewidth=1)

			# Draw line in src image (left)
			a, b, c = lines_src[idx]
			# Handle vertical lines
			if abs(b) > 1e-6:
					x_0, y_0 = 0, -c/b
					x_1, y_1 = width, -(c + a*width)/b
			else:
					# Vertical line
					x_0, x_1 = -c/a, -c/a
					y_0, y_1 = 0, height

			ax[0].plot([x_0, x_1], [y_0, y_1], c=color, alpha=0.5, linewidth=1)

	# Draw points
	ax[0].scatter(src_inliers[:len(src_inliers), 0],
								src_inliers[:len(src_inliers), 1],
								s=10, marker='o', c='lime', edgecolors='black', linewidth=0.5)

	ax[1].scatter(dst_inliers[:len(src_inliers), 0],
								dst_inliers[:len(src_inliers), 1],
								s=10, marker='o', c='lime', edgecolors='black', linewidth=0.5)

	# Show images
	ax[0].imshow(img_src)
	ax[0].set_title(f'Image 1 - {len(src_inliers)} correspondences')
	ax[0].axis('off')

	ax[1].imshow(img_dst)
	ax[1].set_title(f'Image 2 - {len(src_inliers)} correspondences')
	ax[1].axis('off')

	plt.tight_layout()
	plt.show()

def analyze_image_pairs(num_images, keypoints, descriptors, K, gray_images, og_images):
	results = []

	for i in range(num_images - 1):
			print(f"\n{'='*50}")
			print(f"Processing pair ({i}, {i+1})")

			src_pts, dst_pts, good_matches = sift.match_features(i, i+1, keypoints, descriptors)

			print(f"Total matches: {len(good_matches)}")

			# Compute F with my functions
			F_custom, F_mask = F_E_matrices.compute_fundamental_matrix_ransac(src_pts, dst_pts, 1.0, 2000)

			F_mask = F_mask.astype(np.uint8).reshape(-1, 1)
			src_inliers_F = src_pts[F_mask.ravel() == 1]
			dst_inliers_F = dst_pts[F_mask.ravel() == 1]

			F_inlier_ratio = np.sum(F_mask) / len(F_mask)
			print(f"Inliers: {np.sum(F_mask)} / {len(F_mask)} = {F_inlier_ratio:.2%}")

			# Compute E using my functions and the F matrix
			E_custom = F_E_matrices.compute_essential_matrix(F_custom, K, 1.0)

			# Draw epipolar lines
			draw_epipolar_lines(src_inliers_F, dst_inliers_F, F_custom, og_images[i], og_images[i+1])


og_images, gray_images, num_images = ri.read_imgs_from_folder("buddha", show_images=False)
keypoints_, descriptors_ = sift.detect_keypoints_sift(gray_images, num_images, show_images=False)
keypoints_sf, descriptors_sf = sift.apply_nms(keypoints_, descriptors_, og_images)
analyze_image_pairs(num_images, keypoints_sf, descriptors_sf, F_E_matrices.K, gray_images, og_images)