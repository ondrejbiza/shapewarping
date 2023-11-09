import copy
from typing import Dict

import numpy as np
import numpy.typing as npt
import open3d as o3d
import plotly.graph_objects as go


def show_pcd_plotly(
    pcd: npt.NDArray, center: bool = False, axis_visible: bool = True
) -> None:
    """Shows a point cloud using the plotly library."""
    if center:
        pcd = pcd - np.mean(pcd, axis=0, keepdims=True)
    lmin = np.min(pcd)
    lmax = np.max(pcd)

    data = [
        go.Scatter3d(
            x=pcd[:, 0],
            y=pcd[:, 1],
            z=pcd[:, 2],
            marker={"size": 5, "color": pcd[:, 2], "colorscale": "Plotly3"},
            mode="markers",
            opacity=1.0,
        )
    ]
    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout)
    fig.show()


def show_pcds_plotly(
    pcds: Dict[str, npt.NDArray], center: bool = False, axis_visible: bool = True
) -> None:
    colorscales = [
        "Plotly3",
        "Viridis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
    ]

    if center:
        tmp = np.concatenate(list(pcds.values()), axis=0)
        m = np.mean(tmp, axis=0)
        pcds = copy.deepcopy(pcds)
        for k in pcds.keys():
            pcds[k] = pcds[k] - m[None]

    tmp = np.concatenate(list(pcds.values()), axis=0)
    lmin = np.min(tmp)
    lmax = np.max(tmp)

    data = []
    for idx, key in enumerate(pcds.keys()):
        v = pcds[key]
        colorscale = colorscales[idx % len(colorscales)]
        pl = go.Scatter3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            marker={"size": 5, "color": v[:, 2], "colorscale": colorscale},
            mode="markers",
            opacity=1.0,
            name=key,
        )
        data.append(pl)

    layout = {
        "xaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "yaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "zaxis": {"visible": axis_visible, "range": [lmin, lmax]},
        "aspectratio": {"x": 1, "y": 1, "z": 1},
    }

    fig = go.Figure(data=data)
    fig.update_layout(scene=layout, showlegend=True)
    fig.show()


def save_o3d_pcd(pcd: npt.NDArray[np.float32], save_path: str):
    """Saves a point cloud using Open3D.

    Args:
        pcd: Point cloud.
        save_path: Save path with an extension like .pcd.
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(save_path, pcd_o3d)
