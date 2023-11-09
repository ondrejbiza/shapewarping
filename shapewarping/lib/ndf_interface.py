import copy as cp
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import pybullet as pb
import torch

from shapewarping.lib import demo, utils, viz_utils
from shapewarping.lib.object_warping import (
    PARAM_1,
    ObjectWarpingSE2Batch,
    ObjectWarpingSE3Batch,
    warp_to_pcd_se2,
    warp_to_pcd_se3,
)


@dataclass
class NDFInterface:
    """Interface between my method and the Relational Neural Descriptor Fields code."""

    canon_source_path: str = "data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl"
    canon_target_path: str = "data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl"
    canon_source_scale: float = 1.0
    canon_target_scale: float = 1.0
    pcd_subsample_points: Optional[int] = 2000
    nearby_points_delta: float = 0.03
    wiggle: bool = False
    ablate_no_warp: bool = False
    ablate_no_scale: bool = False
    ablate_no_pose_training: bool = False
    ablate_no_size_reg: bool = False

    def __post_init__(self):
        self.canon_source = utils.CanonObj.from_pickle(self.canon_source_path)
        self.canon_target = utils.CanonObj.from_pickle(self.canon_target_path)

    def set_demo_info(
        self,
        pc_master_dict,
        demo_idx: int = 0,
        calculate_cost: bool = False,
        show: bool = True,
    ):
        """Process a demonstration.

        Args:
            pc_master_dict: Dictionary with many demonstrations from R-NDF.
            demo_idx: Index of the demonstration to process.
            calculate_cost: Calculate how well we can fit to the demonstration.
            show: Visualize the demonstration.

        Returns:
            Optionally return a cost of our fit to the demonstration.
        """
        # Get a single demonstration.
        source_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        source_start = np.array(
            pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64
        )
        source_final = np.array(
            pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64
        )

        source_start_pos, source_start_quat = source_start[:3], source_start[3:]
        source_final_pos, source_final_quat = source_final[:3], source_final[3:]
        source_start_trans = utils.pos_quat_to_transform(
            source_start_pos, source_start_quat
        )
        source_final_trans = utils.pos_quat_to_transform(
            source_final_pos, source_final_quat
        )
        source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

        target_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        if (
            self.pcd_subsample_points is not None
            and len(source_pcd) > self.pcd_subsample_points
        ):
            source_pcd, _ = utils.farthest_point_sample(
                source_pcd, self.pcd_subsample_points
            )
        if (
            self.pcd_subsample_points is not None
            and len(target_pcd) > self.pcd_subsample_points
        ):
            target_pcd, _ = utils.farthest_point_sample(
                target_pcd, self.pcd_subsample_points
            )

        # Perception.
        inference_kwargs = {
            "train_latents": not self.ablate_no_warp,
            "train_scales": not self.ablate_no_scale,
            "train_poses": not self.ablate_no_pose_training,
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        warp = ObjectWarpingSE2Batch(
            self.canon_source,
            source_pcd,
            torch.device("cuda:0"),
            **param_1,
            init_scale=self.canon_source_scale,
        )
        source_pcd_complete, _, source_param = warp_to_pcd_se2(
            warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs
        )

        warp = ObjectWarpingSE2Batch(
            self.canon_target,
            target_pcd,
            torch.device("cuda:0"),
            **param_1,
            init_scale=self.canon_target_scale,
        )
        target_pcd_complete, _, target_param = warp_to_pcd_se2(
            warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs
        )

        if show:
            viz_utils.show_pcds_plotly(
                {"pcd": source_pcd, "warp": source_pcd_complete}, center=False
            )
            viz_utils.show_pcds_plotly(
                {"pcd": target_pcd, "warp": target_pcd_complete}, center=False
            )

        # Move object to final pose.
        trans = utils.pos_quat_to_transform(source_param.position, source_param.quat)
        trans = source_start_to_final @ trans
        pos, quat = utils.transform_to_pos_quat(trans)
        source_param.position = pos
        source_param.quat = quat

        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            source_pb, source_param.position, source_param.quat
        )

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            target_pb, target_param.position, target_param.quat
        )

        self.knns, self.deltas, self.target_indices = demo.save_place_nearby_points_v2(
            source_pb,
            target_pb,
            self.canon_source,
            source_param,
            self.canon_target,
            target_param,
            self.nearby_points_delta,
        )

        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        if calculate_cost:
            # Make a prediction based on the training sample and calculate the distance between it and the ground-truth.
            trans_predicted = self.infer_relpose(source_pcd, target_pcd)
            return utils.pose_distance(trans_predicted, source_start_to_final)

    def infer_relpose(
        self,
        source_pcd: npt.NDArray,
        target_pcd: npt.NDArray,
        se3: bool = False,
        show: bool = True,
    ) -> npt.NDArray:
        """Makes a prediction about the final pose of the source object.

        Args:
            source_pcd: Source point cloud (e.g., a mug).
            target_pcd: Target point cloud (e.g., a tree).
            se3: Reason in SE(3).
            show: Show the results of mesh inference.

        Returns:
            4x4 homogenous matrix that transforms the source object to fit onto the target object.
        """
        if (
            self.pcd_subsample_points is not None
            and len(source_pcd) > self.pcd_subsample_points
        ):
            source_pcd, _ = utils.farthest_point_sample(
                source_pcd, self.pcd_subsample_points
            )
        if (
            self.pcd_subsample_points is not None
            and len(target_pcd) > self.pcd_subsample_points
        ):
            target_pcd, _ = utils.farthest_point_sample(
                target_pcd, self.pcd_subsample_points
            )

        inference_kwargs = {
            "train_latents": not self.ablate_no_warp,
            "train_scales": not self.ablate_no_scale,
            "train_poses": not self.ablate_no_pose_training,
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        if se3:
            warp = ObjectWarpingSE3Batch(
                self.canon_source,
                source_pcd,
                torch.device("cuda:0"),
                **param_1,
                init_scale=self.canon_source_scale,
            )
            source_pcd_complete, _, source_param = warp_to_pcd_se3(
                warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs
            )
        else:
            warp = ObjectWarpingSE2Batch(
                self.canon_source,
                source_pcd,
                torch.device("cuda:0"),
                **param_1,
                init_scale=self.canon_source_scale,
            )
            source_pcd_complete, _, source_param = warp_to_pcd_se2(
                warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs
            )

        warp = ObjectWarpingSE2Batch(
            self.canon_target,
            target_pcd,
            torch.device("cuda:0"),
            **param_1,
            init_scale=self.canon_target_scale,
        )
        target_pcd_complete, _, target_param = warp_to_pcd_se2(
            warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs
        )

        if show:
            viz_utils.show_pcds_plotly(
                {"pcd": source_pcd, "warp": source_pcd_complete}, center=False
            )
            viz_utils.show_pcds_plotly(
                {"pcd": target_pcd, "warp": target_pcd_complete}, center=False
            )

        anchors = self.canon_source.to_pcd(source_param)[self.knns]
        targets_source = np.mean(anchors + self.deltas, axis=1)
        targets_target = self.canon_target.to_pcd(target_param)[self.target_indices]

        # Canonical source obj to canonical target obj.
        trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_source, targets_target)

        trans_s_to_b = utils.pos_quat_to_transform(
            source_param.position, source_param.quat
        )
        trans_t_to_b = utils.pos_quat_to_transform(
            target_param.position, target_param.quat
        )

        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            source_pb, *utils.transform_to_pos_quat(trans_s_to_b)
        )

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            target_pb, *utils.transform_to_pos_quat(trans_t_to_b)
        )

        if self.wiggle:
            # Wiggle the source object out of collision.
            src_pos, src_quat = utils.wiggle(source_pb, target_pb)
            trans_s_to_b = utils.pos_quat_to_transform(src_pos, src_quat)

        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        # Compute relative transform.
        trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
        return trans_s_to_t
