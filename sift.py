import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import read_images as ri

def apply_clahe(gray_images):
  clahe_imgs = []
  clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
  for img in gray_images:
      clahe_imgs.append(clahe.apply(img))

  return clahe_imgs

def detect_keypoints_sift(gray_images, num_images, show_images=True):
    sift = cv2.SIFT_create(
        nfeatures=5000,  # Maximum number of features (0 = no limit)
        nOctaveLayers=5,  # Number of layers in each octave
        contrastThreshold=0.04,  # Contrast threshold for filtering weak features
        edgeThreshold=10,  # Edge threshold
        sigma=1.6  # Sigma of Gaussian
    )

    normalized_images = apply_clahe(gray_images)

    listKeypoints = []
    listDescriptors = []

    for img in range(num_images):
        keypoints, descriptors = sift.detectAndCompute(normalized_images[img], None)
        listKeypoints.append(keypoints)
        listDescriptors.append(descriptors)
    if show_images:
        subplot_rows = math.ceil(math.sqrt(num_images))
        subplot_cols = math.ceil(num_images/subplot_rows)

        _, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(22, 10))
        ax = ax.flatten()

        for img in range(subplot_rows*subplot_cols):
            if img < num_images:
                ax[img].imshow(cv2.drawKeypoints(normalized_images[img], listKeypoints[img], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
                ax[img].set_title(f"Image {img+1} - {len(listKeypoints[img])} keypoints")

            ax[img].axis("off")

        plt.tight_layout()
        plt.show()

    return listKeypoints, listDescriptors

def apply_nms(keypoints, descriptors, imgs):
    keypoints_nms = []
    descriptors_nms = []

    for kps, dcs, img in zip(keypoints, descriptors, imgs):
        binary_image = np.zeros((img.shape[0], img.shape[1]))
        response_list = np.array([kp.response for kp in kps])

        mask = np.flip(np.argsort(response_list))

        point_list = np.rint([kp.pt for kp in kps])[mask].astype(int)

        non_max_suppression_mask = []
        for point, index in zip(point_list, mask):
            if binary_image[point[1], point[0]] == 0:
                non_max_suppression_mask.append(index)
                cv2.circle(binary_image, (point[0], point[1]), 3, 255, -1)

        keypoints_nms.append(np.array(kps)[non_max_suppression_mask])
        descriptors_nms.append(np.array(dcs)[non_max_suppression_mask])

    return keypoints_nms, descriptors_nms

def match_features(src_idx, dst_idx, keypoints, descriptors):
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(descriptors[src_idx],descriptors[dst_idx], k = 2)

  good_matches = []
  src_points = []
  dst_points = []

  for m,n in matches:
      if m.distance < 0.7 * n.distance:
          good_matches.append([m])
          src_points.append(keypoints[src_idx][m.queryIdx].pt)
          dst_points.append(keypoints[dst_idx][m.trainIdx].pt)

  src_points = np.float32(src_points).reshape(-1,1,2)
  dst_points = np.float32(dst_points).reshape(-1,1,2)

  return src_points, dst_points, good_matches


# og_images, gray_images, num_images = ri.read_imgs_from_folder("buddha", show_images=False)
# keypoints_, descriptors_ = detect_keypoints_sift(gray_images, num_images, show_images=True)
# keypoints_sf, descriptors_sf = apply_nms(keypoints_, descriptors_, og_images)