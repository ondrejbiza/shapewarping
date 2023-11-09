import argparse
import copy as cp
import logging
import os
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import trimesh
from cycpd import deformable_registration
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from shapewarping.lib import utils, viz_utils

logger = logging.getLogger()


def cpd_transform(
    source: npt.NDArray, target: npt.NDArray, alpha: float = 2.0
) -> npt.NDArray:
    """Warps source points to target points using Coherent Point Drift.

    Args:
        source: Source point cloud of shape [N, 3].
        target: Target point cloud of shape [M, 3].
        alpha: Strength of coherence regularization.

    Returns:
        Translation for each point in source with shape [N*3].
    """

    if len(source.shape) != 2 or source.shape[1] != 3:
        raise ValueError("source should be of shape [N, 3].")

    if len(target.shape) != 2 or target.shape[1] != 3:
        raise ValueError("target should be of shape [M, 3].")

    source, target = source.astype(np.float64), target.astype(np.float64)
    reg = deformable_registration(
        **{"X": source, "Y": target, "tolerance": 0.00001}, alpha=alpha
    )
    reg.register()

    warp = np.dot(reg.G, reg.W)
    return np.hstack(warp)


def warp_gen(
    canonical_index: int,
    objects: List[npt.NDArray],
    scale_factor: float = 1.0,
    alpha: float = 2.0,
    visualize: bool = False,
) -> Tuple[List[npt.NDArray], List[float]]:
    """
    Generates warps from the canonical object to each of the training objects.

    Args:
        canonical_index: Canonical object index.
        objects: Point clouds of training objects.
        scale_factor: Scaling factor applied before warping.
        alpha: Strength of coherence regularization.
        visualize: Show all warps.

    Returns:
        List of warps and their costs.
    """

    logging.info("Selecting canonical object by exhaustive warping.")

    if canonical_index < 0 or canonical_index >= len(objects):
        raise ValueError(f"Invalid canonical index {canonical_index}.")

    source = objects[canonical_index] * scale_factor
    targets = []
    for obj_idx, obj in enumerate(objects):
        if obj_idx != canonical_index:
            targets.append(obj * scale_factor)

    warps = []
    costs = []

    for target_idx, target in enumerate(targets):
        logging.info(f"Computing warps for object {target_idx}.")

        warp = cpd_transform(target, source, alpha=alpha)
        warped = source + warp.reshape(-1, 3)

        costs.append(utils.chamfer_distance_batch(warped, target))
        warps.append(warp)

        if visualize:
            viz_utils.show_pcds_plotly(
                {
                    "target": target,
                    "warp": source + warp.reshape(-1, 3),
                },
                center=True,
            )

    return warps, costs


def pick_canonical_simple(points: List[npt.NDArray]) -> int:
    """Picks a canonical object by computing Chamfer distance between all pairs.

    This functions computes N * (N - 1) Chamfer distances and picks the object
        that is the closest to all other training objects in terms of
        Chamfer distance.

    Args:
        points: List of point clouds for each training object.

    Returns:
        Index of the selected canonical object.
    """

    logging.info("Selecting canonical object by Chamfer distances.")

    if len(points) < 2:
        raise ValueError("We expect at least two point clouds.")

    if torch.has_cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    # GPU acceleration makes this at least 100 times faster.
    points = [torch.tensor(x, device=device, dtype=torch.float32) for x in points]

    overall_costs = []
    for i in range(len(points)):
        logging.info(f"Computing distances for object {i}.")

        cost_per_target = []
        for j in range(len(points)):
            if i != j:
                with torch.no_grad():
                    cost = utils.chamfer_distance_batch_pt(
                        points[i][None], points[j][None]
                    ).cpu()

                cost_per_target.append(cost.item())

        overall_costs.append(np.mean(cost_per_target))

    logger.info(f"All costs: {str(overall_costs)}.")
    return int(np.argmin(overall_costs))


def pick_canonical_warp(points: List[npt.NDArray], alpha: float = 2.0) -> int:
    """Picks a canonical object by searching all possible warps.

    This functions creates N * (N - 1) warps and picks the object
        that can be warped to all other training objects with the lowest
        Chamfer distance.

    Args:
        points: List of point clouds for each training object.
        alpha: Strength of coherence regularization.

    Returns:
        Index of the selected canonical object.
    """

    if len(points) < 2:
        raise ValueError("We expect at least two point clouds.")

    cost_sums = []
    for i in range(len(points)):
        _, costs = warp_gen(i, points, alpha=alpha)
        cost_sums.append(np.sum(costs))
    return int(np.argmin(cost_sums))


def pca_fit_transform(
    distances: npt.NDArray, n_dimensions: int = 8
) -> Tuple[npt.NDArray, PCA]:
    """Fits PCA and transforms data."""

    pca = PCA(n_components=n_dimensions)
    p_components = pca.fit_transform(np.array(distances))
    return p_components, pca


def main(args: argparse.Namespace):
    np.random.seed(2023)
    trimesh.util.attach_to_log()

    obj_paths = [os.path.join(args.load_dir, x) for x in os.listdir(args.load_dir)]
    rotation = Rotation.from_euler(
        "zyx", [args.rot_z, args.rot_y, args.rot_x]
    ).as_matrix()
    num_surface_samples = args.num_surface_samples

    meshes = []
    for obj_path in obj_paths:
        mesh = utils.trimesh_load_object(obj_path)
        utils.trimesh_transform(mesh, center=True, scale=None, rotation=rotation)
        meshes.append(mesh)

    # TODO: Simplify this.
    small_surface_points = []
    surface_points = []
    mesh_points = []
    hybrid_points = []
    faces = []

    for mesh in meshes:
        print("surface")
        sp = utils.trimesh_create_verts_surface(
            mesh, num_surface_samples=num_surface_samples
        )
        print("x")
        ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
        print("y")
        mp, f = utils.trimesh_get_vertices_and_faces(mesh)
        print("scale mesh")
        ssp, sp, mp = utils.scale_points_circle([ssp, sp, mp], base_scale=0.1)
        h = np.concatenate([mp, sp])  # Order important!

        small_surface_points.append(ssp)
        surface_points.append(sp)
        mesh_points.append(mp)
        faces.append(f)
        hybrid_points.append(h)

    print("####### Pick canonical shape")
    # TODO: Blacklist objects that have too many vertices.
    if args.pick_canon_warp:
        canonical_idx = pick_canonical_warp(small_surface_points)
    else:
        # TODO: Should we be using hybrid points.
        canonical_idx = pick_canonical_simple(hybrid_points)

    # We use small point clouds, except for the canonical object, to figure out the warps.
    tmp_obj_points = cp.copy(small_surface_points)
    tmp_obj_points[canonical_idx] = hybrid_points[canonical_idx]

    print("######## Create warps")
    warps, _ = warp_gen(
        canonical_idx, tmp_obj_points, alpha=args.alpha, visualize=args.show
    )
    print("######## Fit PCA")
    _, pca = pca_fit_transform(warps, n_dimensions=args.n_dimensions)

    with open(args.save_path, "wb") as f:
        pickle.dump(
            {
                "pca": pca,
                "canonical_obj": hybrid_points[canonical_idx],
                "canonical_mesh_points": mesh_points[canonical_idx],
                "canonical_mesh_faces": faces[canonical_idx],
            },
            f,
        )


if __name__ == "__main__":
    # TODO: Set good defaults.
    parser = argparse.ArgumentParser("Learns a latent space of object warps.")

    parser.add_argument("load_dir", help="Directory with training meshes.")
    parser.add_argument("save_path", help="Save path for pickled model.")

    parser.add_argument(
        "--rot-x", type=float, default=None, help="x-axis rotation in radians."
    )
    parser.add_argument(
        "--rot-y", type=float, default=None, help="y-axis rotation in radians."
    )
    parser.add_argument(
        "--rot-z", type=float, default=None, help="z-axis rotation in radians."
    )

    parser.add_argument(
        "--num-surface-samples",
        type=int,
        default=10000,
        help="Number of points to sample on the surface of each mesh.",
    )

    parser.add_argument(
        "--n-dimensions", type=int, default=8, help="Number of PCA dimensions."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Strength of coherence regularization. The lower, the more warping we do.",
    )
    parser.add_argument(
        "--pick-canon-warp",
        default=False,
        action="store_true",
        help="Perform object warping when we pick the canonical object.",
    )
    parser.add_argument("--show", default=False, action="store_true")
    main(parser.parse_args())
