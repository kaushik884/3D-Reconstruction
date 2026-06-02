# Kaushik Deo
# April 23 2026
"""Monocular depth estimation using Depth Anything V2."""
from __future__ import annotations
import os
from typing import List, Optional, Sequence
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from .io_utils import list_images


def generate_depth_maps(
    input_dir: str,
    output_dir: str,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Optional[str] = None,
    filenames: Optional[Sequence[str]] = None,
    overwrite: bool = False,
) -> List[str]:
    raw_dir = os.path.join(output_dir, "raw_npy")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[Depth] Using device: {dev}")

    files = list(filenames) if filenames is not None else list_images(input_dir)
    # Skip images that already have a cached depth map unless overwrite is requested
    if not overwrite:
        remaining = [
            f for f in files
            if not os.path.exists(os.path.join(raw_dir, f"{os.path.splitext(f)[0]}_depth.npy"))
        ]
        if len(remaining) < len(files):
            print(f"[Depth] Skipping {len(files) - len(remaining)} cached depth maps")
        files = remaining

    if not files:
        print("[Depth] All depth maps already exist; nothing to do.")
        return []

    # Load pretrained Depth Anything V2 from HuggingFace (monocular depth estimator)
    print(f"[Depth] Loading {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(dev)
    model.eval()

    written = []
    with torch.no_grad():
        for i, name in enumerate(files, 1):
            img_path = os.path.join(input_dir, name)
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            inputs = processor(images=image, return_tensors="pt").to(dev)
            outputs = model(**inputs)

            # Upsample network output back to the original image resolution
            prediction = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(H, W), mode="bicubic", align_corners=False,
            ).squeeze().cpu().numpy()

            base = os.path.splitext(name)[0]
            npy_path = os.path.join(raw_dir, f"{base}_depth.npy")
            np.save(npy_path, prediction)
            written.append(npy_path)

            # Normalize + inferno colormap for human-readable depth previews
            dmin, dmax = prediction.min(), prediction.max()
            norm = (prediction - dmin) / (dmax - dmin + 1e-9)
            colored = (plt.get_cmap("inferno")(norm)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(vis_dir, f"{base}_depth_vis.png"),
                        cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

            if i % 5 == 0 or i == len(files):
                print(f"[Depth] Processed {i}/{len(files)}")
    print(f"[Depth] Wrote {len(written)} depth maps to {raw_dir}")
    return written
