import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go


def show_pcd_plotly(pcd: npt.NDArray, center: bool=False, axis_visible: bool=True) -> None:
    """Shows a point cloud using the plotly library."""

    if center:
        pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    lmin = np.min(pcd)
    lmax = np.max(pcd)

    data = [go.Scatter3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], marker={"size": 5, "color": pcd[:, 2], "colorscale": "Plotly3"}, mode="markers", opacity=1.)]
    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1}
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout)
    fig.show()
