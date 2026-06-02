# Kaushik Deo
# April 23 2026
"""
Examples:
    # Full pipeline on datasets
    python scripts/main.py --dataset dtu_scan114 --stages all

    # Sparse SfM only, skipping BA
    python scripts/main.py --dataset buddha --stages sparse --no-ba
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.config import Config, DATASET_PRESETS
from pipeline import io_utils, features, epipolar, sparse, bundle_adj, depth, dense, classify, viz


# Ordered pipeline stages: two-view geometry -> SfM -> monocular depth -> dense cloud -> classification
STAGES = ["epipolar", "sparse", "depth", "dense", "classify"]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="3D reconstruction pipeline")
    p.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()),
                   default="dtu_scan114", help="Preset dataset name")
    p.add_argument("--images-dir", default=None, help="Override: path to image folder")
    p.add_argument("--cams-dir", default=None, help="Override: MVSNet cams_1 folder")
    p.add_argument("--name", default=None, help="Output run name (defaults to dataset name)")
    p.add_argument("--output-dir", default="outputs", help="Base outputs directory")
    p.add_argument("--stages", default="sparse,depth,dense,classify",
                   help="Comma-separated subset of {epipolar,sparse,depth,dense,classify} or 'all'")
    p.add_argument("--no-ba", action="store_true", help="Skip bundle adjustment in the sparse stage")
    p.add_argument("--show", action="store_true", help="Show matplotlib / open3d viewers")
    p.add_argument("--fx", type=float); p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float); p.add_argument("--cy", type=float)
    p.add_argument("--device", default="cuda")
    p.add_argument("--classify-use-rgb", action="store_true")
    args = p.parse_args()

    cfg = Config(dataset=args.dataset)
    if args.images_dir:
        cfg.images_dir = args.images_dir
    if args.cams_dir:
        cfg.cams_dir = args.cams_dir
    if args.name:
        cfg.name = args.name
    cfg.output_dir = args.output_dir
    cfg.run_ba = not args.no_ba
    cfg.show_viz = args.show
    cfg.device = args.device
    cfg.classify_use_rgb = args.classify_use_rgb

    # Build intrinsics matrix K from CLI overrides when all four params are given
    if all(v is not None for v in (args.fx, args.fy, args.cx, args.cy)):
        cfg.K = np.array([[args.fx, 0.0, args.cx],
                          [0.0, args.fy, args.cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if stages == ["all"]:
        stages = ["sparse", "depth", "dense", "classify"]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"Unknown stage: {s!r}. Choose from {STAGES} or 'all'.")
    cfg.stages = stages
    return cfg.resolve()


def _load_sparse_if_needed(cfg: Config, rgb, gray, filenames, K):
    # Re-use cached sparse reconstruction if available; otherwise regenerate it
    cache = cfg.sparse_npz_path()
    if os.path.exists(cache):
        print(f"[Pipeline] Loading cached sparse reconstruction from {cache}")
        data = np.load(cache, allow_pickle=True)
        return data["reconstruction"].item()
    print("[Pipeline] No sparse cache; running sparse stage first")
    return _run_sparse(cfg, rgb, gray, filenames, K)


def _run_epipolar(cfg: Config, rgb, gray, filenames, K) -> None:
    # Epipolar stage: visualize two-view geometry for consecutive image pairs
    keypoints, descriptors = features.detect_keypoints(
        gray, cfg.sift_contrast_threshold, cfg.sift_edge_threshold
    )
    out_dir = os.path.join(cfg.run_dir(), "epipolar")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(gray) - 1):
        src, dst, _, _ = features.match_pair(
            i, i + 1, keypoints, descriptors, cfg.match_ratio, cfg.device
        )
        # 8-point algorithm minimum for fundamental matrix estimation
        if len(src) < 8:
            print(f"[Epipolar] Pair ({i}, {i+1}): not enough matches ({len(src)})")
            continue
        F, _, src_in, dst_in = epipolar.compute_fundamental(
            src, dst, cfg.ransac_threshold_px, cfg.ransac_max_iters
        )
        print(f"[Epipolar] Pair ({i},{i+1}): {len(src_in)}/{len(src)} inliers, F=\n{F}")
        epipolar.draw_epipolar_lines(
            src_in, dst_in, F, rgb[i], rgb[i + 1],
            save_path=os.path.join(out_dir, f"epipolar_{i:03d}_{i+1:03d}.png"),
            show=cfg.show_viz,
        )


def _run_sparse(cfg: Config, rgb, gray, filenames, K) -> dict:
    keypoints, descriptors = features.detect_keypoints(
        gray, cfg.sift_contrast_threshold, cfg.sift_edge_threshold
    )
    print("[Pipeline] Matching all pairs")
    all_matches = features.match_all_pairs(
        keypoints, descriptors, cfg.match_ratio, cfg.device
    )
    print(f"[Pipeline] {len(all_matches)} pairs with matches")

    # Incremental SfM: initialize with best pair, then register remaining cameras via PnP
    reconstruction = sparse.run_sparse_sfm(keypoints, all_matches, K, cfg)
    # Refine all camera poses and 3D points jointly to minimize reprojection error
    if cfg.run_ba:
        reconstruction = bundle_adj.run_bundle_adjustment(reconstruction, K)

    np.savez(cfg.sparse_npz_path(), reconstruction=np.array(reconstruction, dtype=object))
    print(f"[Pipeline] Saved sparse reconstruction to {cfg.sparse_npz_path()}")

    viz.visualize_sparse(
        reconstruction,
        save_html=os.path.join(cfg.run_dir(), f"{cfg.name}_sparse.html"),
        show=cfg.show_viz,
    )
    return reconstruction


def _run_depth(cfg: Config, filenames) -> None:
    depth.generate_depth_maps(
        cfg.images_dir, cfg.depth_dir(), cfg.depth_model, cfg.device, filenames
    )


def _run_dense(cfg: Config, rgb, gray, filenames, K) -> str:
    reconstruction = _load_sparse_if_needed(cfg, rgb, gray, filenames, K)
    # Make sure depth maps exist
    depth.generate_depth_maps(cfg.images_dir, cfg.depth_dir(), cfg.depth_model, cfg.device, filenames)

    # Rescale monocular depth maps so they match the metric scale from SfM
    aligned = dense.align_depth_maps(reconstruction, cfg.depth_dir(), filenames)
    if not aligned:
        raise RuntimeError("No depth maps could be aligned.")

    # Unproject every pixel to 3D and keep only those consistent across multiple views
    points, colors = dense.unproject_with_consistency(
        aligned, reconstruction, K, rgb,
        max_depth=cfg.dense_max_depth,
        depth_tol=cfg.dense_depth_tol,
        min_consistent_views=cfg.dense_min_consistent_views,
        device=cfg.device,
    )
    ply_path = cfg.dense_ply_path()
    dense.build_and_save_point_cloud(points, colors, ply_path, cfg.dense_cleanup, cfg.show_viz)
    return ply_path


def _run_classify(cfg: Config) -> None:
    ply = cfg.dense_ply_path()
    if not os.path.exists(ply):
        raise FileNotFoundError(f"Dense PLY not found at {ply}; run --stages dense first.")
    classify.classify_ply(
        ply,
        use_rgb=cfg.classify_use_rgb,
        num_points=cfg.classify_num_points,
        show=cfg.show_viz,
    )


def main() -> None:
    cfg = parse_args()
    print(f"=== 3D Reconstruction pipeline ===")
    print(f"  dataset: {cfg.dataset}")
    print(f"  images:  {cfg.images_dir}")
    print(f"  stages:  {cfg.stages}")
    print(f"  run_dir: {cfg.run_dir()}")

    # Load images once and resolve a single intrinsic matrix K for the whole run
    rgb, gray, filenames = io_utils.read_images(cfg.images_dir)
    K = io_utils.resolve_intrinsics(cfg.cams_dir, filenames, cfg.K)
    print(f"  K =\n{K}")

    for stage in cfg.stages:
        print(f"\n{'='*60}\nSTAGE: {stage}\n{'='*60}")
        if stage == "epipolar":
            _run_epipolar(cfg, rgb, gray, filenames, K)
        elif stage == "sparse":
            _run_sparse(cfg, rgb, gray, filenames, K)
        elif stage == "depth":
            _run_depth(cfg, filenames)
        elif stage == "dense":
            _run_dense(cfg, rgb, gray, filenames, K)
        elif stage == "classify":
            _run_classify(cfg)

    print("\n[Pipeline] Done.")


if __name__ == "__main__":
    main()
