from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pybullet as pb

from shapewarping.lib import utils, viz_utils


def save_pick_contact_points(robotiq_id: int, source_id: int, trans_robotiq_to_ws: NDArray,
                             canon_source: utils.CanonObj, source_param: utils.ObjParam) -> Tuple[NDArray, NDArray[np.int32]]:

    pb.performCollisionDetection()
    cols = pb.getClosestPoints(robotiq_id, source_id, 0.0)

    pos_robotiq = [col[5] for col in cols]
    pos_source = [col[6] for col in cols]
    print("# contact points:", len(pos_robotiq))

    assert len(pos_robotiq) > 0

    pos_robotiq = np.stack(pos_robotiq, axis=0).astype(np.float32)
    pos_source = np.stack(pos_source, axis=0).astype(np.float32)

    # Contact points on the gripper in its base position.
    pos_robotiq_canon = utils.transform_pcd(pos_robotiq, np.linalg.inv(trans_robotiq_to_ws))

    # Contact point indices on the source object.
    trans_source_to_ws = source_param.get_transform()
    pos_source_canon = utils.transform_pcd(pos_source, np.linalg.inv(trans_source_to_ws))
    source_pcd_complete = canon_source.to_pcd(source_param)

    dist = np.sqrt(np.sum(np.square(source_pcd_complete[:, None] - pos_source_canon[None]), axis=2))
    index = np.argmin(dist, axis=0).transpose()

    return pos_robotiq_canon, index


def get_knn_and_deltas(obj: NDArray, vps: NDArray, k: int=10,
                       show: bool=False) -> Tuple[NDArray, NDArray]:
    """Anchor virtual points on an object using k-nearest-neighbors."""
    if show:
        viz_utils.show_pcds_pyplot({
            "obj": obj,
            "vps": vps
        })

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        # Get K closest points, compute relative vectors.
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list


def save_place_nearby_points(source: int, target: int, canon_source_obj: utils.CanonObj,
                             source_obj_param: utils.ObjParam, canon_target_obj: utils.CanonObj,
                             target_obj_param: utils.ObjParam, delta: float,
                             draw_spheres: bool=False
                             ) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    """Process demonstration by setting up warping of nearby points.
    
    We find points on the target object that are nearby the source object.
    Then we define virtual points on the source object which are anchored
    to it using k-nearest-neighbors."""
    pb.performCollisionDetection()
    cols = pb.getClosestPoints(source, target, delta)

    if draw_spheres:
        # Add spheres at contact points to pybullet.
        for col in cols:
            pos_mug = col[5]
            s = pb.loadURDF("data/sphere_red.urdf")
            utils.pb_set_pose(s, pos_mug, np.array([0., 0., 0., 1.]))

            pos_tree = col[6]
            s = pb.loadURDF("data/sphere.urdf")
            utils.pb_set_pose(s, pos_tree, np.array([0., 0., 0., 1.]))

    pos_source = [col[5] for col in cols]
    pos_target = [col[6] for col in cols]

    assert len(pos_source) > 0

    # Pairs of nearby points:
    # Points on source object in world coordinates.
    pos_source = np.stack(pos_source, axis=0).astype(np.float32)
    # Points on target object in world coordinates.
    pos_target = np.stack(pos_target, axis=0).astype(np.float32)

    # Points on target in canonical target object coordinates.
    pos_target_target_coords = utils.transform_pcd(pos_target, np.linalg.inv(target_obj_param.get_transform()))
    # Points on target in canonical source object coordinates.
    pos_target_source_coords = utils.transform_pcd(pos_target, np.linalg.inv(source_obj_param.get_transform()))

    full_source_pcd = canon_source_obj.to_pcd(source_obj_param)
    full_target_pcd = canon_target_obj.to_pcd(target_obj_param)

    knns, deltas = get_knn_and_deltas(full_source_pcd, pos_target_source_coords)

    dist_2 = np.sqrt(np.sum(np.square(full_target_pcd[:, None] - pos_target_target_coords[None]), axis=2))
    i_2 = np.argmin(dist_2, axis=0).transpose()

    return knns, deltas, i_2


def save_place_nearby_points_v2(source: int, target: int, canon_source_obj: utils.CanonObj,
                                source_obj_param: utils.ObjParam, canon_target_obj: utils.CanonObj,
                                target_obj_param: utils.ObjParam, delta: float,
                                draw_spheres: bool=False
                                ) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:

    source_pcd = canon_source_obj.to_transformed_pcd(source_obj_param)
    target_pcd = canon_target_obj.to_transformed_pcd(target_obj_param)

    dist = np.sqrt(np.sum(np.square(source_pcd[:, None] - target_pcd[None]), axis=-1))
    print("@@", np.min(dist))
    indices = np.where(dist <= delta)
    pos_source = source_pcd[indices[0]]
    pos_target = target_pcd[indices[1]]

    assert len(pos_source) > 0, "No nearby points in demonstration."
    print("# nearby points:", len(pos_source))
    if len(pos_source) < 10:
        print("WARNING: Too few nearby points.")

    max_pairs = 100
    if len(pos_source) > max_pairs:
        pos_source, indices2 = utils.farthest_point_sample(pos_source, max_pairs)
        pos_target = pos_target[indices2]

    # Points on target in canonical target object coordinates.
    pos_target_target_coords = utils.transform_pcd(pos_target, np.linalg.inv(target_obj_param.get_transform()))
    # Points on target in canonical source object coordinates.
    pos_target_source_coords = utils.transform_pcd(pos_target, np.linalg.inv(source_obj_param.get_transform()))

    full_source_pcd = canon_source_obj.to_pcd(source_obj_param)
    full_target_pcd = canon_target_obj.to_pcd(target_obj_param)

    knns, deltas = get_knn_and_deltas(full_source_pcd, pos_target_source_coords)

    dist_2 = np.sqrt(np.sum(np.square(full_target_pcd[:, None] - pos_target_target_coords[None]), axis=2))
    i_2 = np.argmin(dist_2, axis=0).transpose()

    return knns, deltas, i_2
