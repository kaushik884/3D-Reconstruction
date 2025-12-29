import plotly.graph_objects as go
import numpy as np

def visualize_reconstruction(reconstruction):
    # Convert points_3d to numpy array if needed
    if isinstance(reconstruction['points_3d'], list):
        points_3d = np.array(reconstruction['points_3d'])
    else:
        points_3d = reconstruction['points_3d']

    # Extract camera positions
    camera_positions = []
    for cam_idx, cam_data in reconstruction['cameras'].items():
        R = cam_data['R']
        t = cam_data['t']
        # Camera center: C = -R^T @ t
        camera_center = -R.T @ t
        camera_positions.append(camera_center.ravel())

    camera_positions = np.array(camera_positions)

    # Create figure
    fig = go.Figure()

    # Add 3D points
    fig.add_trace(go.Scatter3d(x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2], mode='markers',
                  marker=dict(size=1, color='blue', opacity=0.3), name='Points'))

    # Add cameras
    fig.add_trace(go.Scatter3d(x=camera_positions[:, 0], y=camera_positions[:, 1], z=camera_positions[:, 2], mode='markers',
        marker=dict(size=6, color='red'), name='Cameras'))

    # Simple layout
    fig.update_layout(
        title=f'{len(reconstruction["cameras"])} cameras, {len(points_3d)} points',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=800,
        height=600,
        showlegend=True
    )

    return fig
