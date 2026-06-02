# Kaushik Deo
# April 23 2026
"""PointNet++ classification of the reconstructed point cloud."""
from __future__ import annotations
import os
import sys
from typing import Optional, Tuple
import numpy as np
import open3d as o3d
import torch

MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Pointnet_Pointnet2_pytorch"))
MODELS_DIR = os.path.join(REPO_DIR, "models")
for p in (REPO_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.append(p)

DEFAULT_WEIGHTS = os.path.join(REPO_DIR, "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth")


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    num_points: int = 1024,
    show: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (tensor_xyz [1,3,N], tensor_xyz_rgb [1,6,N])."""
    print(f"[Classify] Preprocessing cloud with {len(pcd.points)} pts")
    # Farthest-point sampling preserves shape while giving PointNet++ a fixed-size input
    down = pcd.farthest_point_down_sample(num_points)

    # Center at origin and scale into unit sphere (ModelNet40 training convention)
    xyz = np.asarray(down.points)
    centroid = xyz.mean(axis=0)
    xyz_c = xyz - centroid
    max_d = np.max(np.linalg.norm(xyz_c, axis=1))
    xyz_n = xyz_c / max(max_d, 1e-9)

    rgb = np.asarray(down.colors) if down.has_colors() else np.zeros_like(xyz_n)

    t_xyz = torch.tensor(xyz_n.T, dtype=torch.float32).unsqueeze(0)
    t_xyz_rgb = torch.tensor(np.hstack([xyz_n, rgb]).T, dtype=torch.float32).unsqueeze(0)

    if show:
        o3d.visualization.draw_geometries([down], window_name=f"{num_points} FPS points")
    return t_xyz, t_xyz_rgb


def classify(
    tensor_input: torch.Tensor,
    use_rgb: bool = False,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    top_k: int = 3,
) -> Optional[Tuple[str, float]]:
    from pointnet2_cls_ssg import get_model  # type: ignore

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Classify] Inference device: {dev}")

    weights = weights_path or DEFAULT_WEIGHTS
    # Load PointNet++ SSG pretrained on ModelNet40 (40 object classes)
    model = get_model(len(MODELNET40_CLASSES), normal_channel=use_rgb).to(dev)
    checkpoint = torch.load(weights, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tensor_input = tensor_input.to(dev)
    with torch.no_grad():
        logits, _ = model(tensor_input)
        # Convert logits to class probabilities and pick the top-k predictions
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        top_probs, top_idx = torch.topk(probs, top_k)

    mode = "XYZ + RGB" if use_rgb else "XYZ only"
    print(f"\n=== Classification Results ({mode}) ===")
    for i in range(top_k):
        cls = MODELNET40_CLASSES[top_idx[i].item()]
        conf = top_probs[i].item() * 100
        print(f"  {i+1}. {cls:>12s}: {conf:.2f}%")
    return MODELNET40_CLASSES[top_idx[0].item()], top_probs[0].item()


def classify_ply(
    ply_path: str,
    use_rgb: bool = False,
    num_points: int = 1024,
    show: bool = False,
    weights_path: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    pcd = o3d.io.read_point_cloud(ply_path)
    print(ply_path)
    t_xyz, t_xyz_rgb = preprocess_point_cloud(pcd, num_points=num_points, show=show)
    tensor = t_xyz_rgb if use_rgb else t_xyz
    return classify(tensor, use_rgb=use_rgb, weights_path=weights_path)
