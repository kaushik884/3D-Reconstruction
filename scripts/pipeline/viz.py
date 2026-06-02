# Kaushik Deo
# April 23 2026
"""3D visualization helpers (plotly for sparse, open3d handled in dense module)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go


def visualize_sparse(reconstruction: Dict, save_html: Optional[str] = None, show: bool = False) -> go.Figure:
    points_3d = reconstruction["points_3d"]
    if not isinstance(points_3d, np.ndarray):
        points_3d = np.array(points_3d)

    # Convert world->camera (R, t) into camera centers in world frame: C = -R^T @ t
    camera_positions = []
    for cam in reconstruction["cameras"].values():
        camera_positions.append((-cam["R"].T @ cam["t"]).ravel())
    camera_positions = np.array(camera_positions)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
        mode="markers", marker=dict(size=1, color="blue", opacity=0.3), name="Points",
    ))
    fig.add_trace(go.Scatter3d(
        x=camera_positions[:, 0], y=camera_positions[:, 1], z=camera_positions[:, 2],
        mode="markers", marker=dict(size=6, color="red"), name="Cameras",
    ))
    fig.update_layout(
        title=f"{len(reconstruction['cameras'])} cameras, {len(points_3d)} points",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=900, height=700, showlegend=True,
    )
    if save_html is not None:
        fig.write_html(save_html)
        print(f"[Viz] Sparse reconstruction saved to {save_html}")
    if show:
        fig.show()
    return fig
