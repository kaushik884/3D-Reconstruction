# Kaushik Deo
# April 23 2026
"""SIFT feature extraction, NMS, and GPU-accelerated descriptor matching."""
from __future__ import annotations
from typing import List, Sequence, Tuple
import cv2
import numpy as np
import torch


def apply_clahe(gray_images: Sequence[np.ndarray]) -> List[np.ndarray]:
    # CLAHE boosts local contrast so SIFT can find features in dim / low-texture regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return [clahe.apply(img) for img in gray_images]


def detect_keypoints(
    gray_images: Sequence[np.ndarray],
    contrast_threshold: float = 0.02,
    edge_threshold: float = 10.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Detect SIFT keypoints on CLAHE-normalized grayscale images and apply NMS."""
    sift = cv2.SIFT_create(
        nfeatures=10000,
        nOctaveLayers=5,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=1.6,
    )
    clahe_imgs = apply_clahe(gray_images)

    keypoints, descriptors = [], []
    for img in clahe_imgs:
        kps, descs = sift.detectAndCompute(img, None)
        keypoints.append(kps)
        descriptors.append(descs)

    # Non-max suppression on a 3-px pixel grid
    keypoints, descriptors = _apply_nms(keypoints, descriptors, clahe_imgs)

    total = sum(len(k) for k in keypoints)
    print(f"Detected {total} keypoints across {len(keypoints)} images")
    return keypoints, descriptors


def _apply_nms(
    keypoints: Sequence[Sequence[cv2.KeyPoint]],
    descriptors: Sequence[np.ndarray],
    imgs: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    kps_out, descs_out = [], []
    for kps, descs, img in zip(keypoints, descriptors, imgs):
        binary = np.zeros(img.shape[:2], dtype=np.uint8)
        # Sort keypoints by response (strongest first) so weaker ones get suppressed
        response = np.array([kp.response for kp in kps])
        order = np.flip(np.argsort(response))
        points = np.rint([kp.pt for kp in kps])[order].astype(int)

        kept = []
        for (x, y), idx in zip(points, order):
            # Accept only if no stronger keypoint has already claimed this 3-px neighborhood
            if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1] and binary[y, x] == 0:
                kept.append(idx)
                cv2.circle(binary, (int(x), int(y)), 3, 255, -1)

        kps_out.append(np.array(kps)[kept])
        descs_out.append(np.array(descs)[kept])
    return kps_out, descs_out


def match_pair(
    src_idx: int,
    dst_idx: int,
    keypoints: Sequence[Sequence[cv2.KeyPoint]],
    descriptors: Sequence[np.ndarray],
    ratio: float = 0.8,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU brute-force matcher with Lowe's ratio test.

    Returns (src_pts[N,1,2], dst_pts[N,1,2], src_kp_idx, dst_kp_idx).
    The keypoint indices let SfM track the same feature across pairs.
    """
    d1 = descriptors[src_idx]
    d2 = descriptors[dst_idx]
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        empty_f = np.empty((0, 1, 2), dtype=np.float32)
        empty_i = np.empty((0,), dtype=np.int64)
        return empty_f, empty_f, empty_i, empty_i

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    t1 = torch.from_numpy(d1).to(dev, dtype=torch.float32)
    t2 = torch.from_numpy(d2).to(dev, dtype=torch.float32)

    # Pairwise L2 distance between every descriptor in image 1 and image 2
    dist = torch.cdist(t1, t2)  # (N, M)
    # Lowe's ratio test: keep match only if best is clearly better than second-best
    vals, idx = dist.topk(2, largest=False, dim=1)
    ratio_mask = vals[:, 0] < ratio * vals[:, 1]

    idx1 = torch.arange(t1.shape[0], device=dev)[ratio_mask].cpu().numpy()
    idx2 = idx[ratio_mask, 0].cpu().numpy()

    src_pts = np.float32([keypoints[src_idx][i].pt for i in idx1]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[dst_idx][i].pt for i in idx2]).reshape(-1, 1, 2)
    return src_pts, dst_pts, idx1.astype(np.int64), idx2.astype(np.int64)


def match_all_pairs(
    keypoints: Sequence[Sequence[cv2.KeyPoint]],
    descriptors: Sequence[np.ndarray],
    ratio: float = 0.75,
    device: str = "cuda",
) -> dict:
    """Match every (i, j) pair with i < j. Returns dict keyed by (i, j)."""
    n = len(keypoints)
    all_matches = {}
    for i in range(n):
        for j in range(i + 1, n):
            src, dst, ki, kj = match_pair(i, j, keypoints, descriptors, ratio, device)
            if len(src) > 0:
                all_matches[(i, j)] = {
                    "src_pts": src,
                    "dst_pts": dst,
                    "src_kp_idx": ki,
                    "dst_kp_idx": kj,
                }
    return all_matches
