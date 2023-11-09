from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import nn, optim

from shapewarping.lib import utils

# Parameters used in my paper.
PARAM_1 = {"lr": 1e-2, "n_steps": 100, "n_samples": 1000, "object_size_reg": 0.01}


class ObjectWarping:
    """Base class for inference of object shape, pose and scale with gradient descent."""

    def __init__(
        self,
        canon_obj: utils.CanonObj,
        pcd: NDArray[np.float32],
        device: torch.device,
        lr: float,
        n_steps: int,
        n_samples: Optional[int] = None,
        object_size_reg: Optional[float] = None,
        init_scale: float = 1.0,
    ):
        """Generic init functions that save the canonical object and the observed point cloud.

        Args:
            canon_obj: Canonical object instance.
            pcd: Target point cloud.
            device: Pytorch device (e.g., "cuda:0", "cpu").
            n_steps: Number of gradient descent steps.
            n_samples: Number of random restarts.
            object_size_reg: Warped object size regularization strength.
            init_scale: Initial warped object scale.
        """
        self.device = device
        self.pca = canon_obj.pca
        self.lr = lr
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.object_size_reg = object_size_reg
        self.init_scale = init_scale

        self.global_means = np.mean(pcd, axis=0)
        pcd = pcd - self.global_means[None]

        self.canonical_pcd = torch.tensor(
            canon_obj.canonical_pcd, dtype=torch.float32, device=device
        )
        self.pcd = torch.tensor(pcd, dtype=torch.float32, device=device)

        if canon_obj.pca is not None:
            self.means = torch.tensor(
                canon_obj.pca.mean_, dtype=torch.float32, device=device
            )
            self.components = torch.tensor(
                canon_obj.pca.components_, dtype=torch.float32, device=device
            )
        else:
            self.means = None
            self.components = None

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        """See subclasses."""
        raise NotImplementedError()

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor, canonical_pcd: torch.Tensor
    ) -> torch.Tensor:
        """See subclasses."""
        raise NotImplementedError()

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """See subclasses."""
        raise NotImplementedError()

    def subsample(
        self, num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly subsamples the canonical object, including its PCA projection."""
        indices = np.random.randint(0, self.components.shape[1] // 3, num_samples)
        means_ = self.means.view(-1, 3)[indices].view(-1)
        components_ = self.components.view(self.components.shape[0], -1, 3)[:, indices]
        components_ = components_.view(self.components.shape[0], -1)
        canonical_obj_pt_ = self.canonical_pcd[indices]

        return means_, components_, canonical_obj_pt_

    def inference(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Runs inference for a batch of initial poses."""
        self.initialize_params_and_opt(
            initial_poses,
            initial_centers,
            initial_latents,
            initial_scales,
            train_latents,
            train_centers,
            train_poses,
            train_scales,
        )

        for _ in range(self.n_steps):
            if self.n_samples is not None:
                means_, components_, canonical_pcd_ = self.subsample(self.n_samples)
            else:
                means_ = self.means
                components_ = self.components
                canonical_pcd_ = self.canonical_pcd

            self.optim.zero_grad()
            new_pcd = self.create_warped_transformed_pcd(
                components_, means_, canonical_pcd_
            )
            cost = cost_batch_pt(self.pcd[None], new_pcd)

            if self.object_size_reg is not None:
                size = torch.max(
                    torch.sqrt(torch.sum(torch.square(new_pcd), dim=-1)), dim=-1
                )[0]
                cost = cost + self.object_size_reg * size

            cost.sum().backward()
            self.optim.step()

        with torch.no_grad():
            # Compute final cost without subsampling.
            new_pcd = self.create_warped_transformed_pcd(
                self.components, self.means, self.canonical_pcd
            )
            cost = cost_batch_pt(self.pcd[None], new_pcd)

            if self.object_size_reg is not None:
                size = torch.max(
                    torch.sqrt(torch.sum(torch.square(new_pcd), dim=-1)), dim=-1
                )[0]
                cost = cost + self.object_size_reg * size

        return self.assemble_output(cost)


class ObjectWarpingSE3Batch(ObjectWarping):
    """Object shape and pose warping in SE3."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        """Initializes warping parameters and optimization.

        Args:
            initial_poses: Array of initial warped object poses.
            initial_centers: Array of initial warped object centers.
            initial_latents: Array of initial warped object latent vectors.
            initial_scales: Array of initial warped object scales.
            train_latents: If False, do not optimize the shape of the object.
            train_centers: If False, do not optimize object centers.
            train_poses: If False, do not optimize object poses.
            train_scales: If False, do not optimize object scale.
        """
        n_angles = len(initial_poses)

        # Initial rotation matrices.
        self.initial_poses = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        # This 2x3 vectors will turn into an identity rotation matrix.
        unit_ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        unit_ortho = np.repeat(unit_ortho[None], n_angles, axis=0)
        init_ortho_pt = torch.tensor(
            unit_ortho, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_latents is None:
            initial_latents_pt = torch.zeros(
                (n_angles, self.pca.n_components),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            initial_latents_pt = torch.tensor(
                initial_latents, dtype=torch.float32, device=self.device
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.latent_param = nn.Parameter(initial_latents_pt, requires_grad=True)
        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(init_ortho_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_latents:
            self.params.append(self.latent_param)
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor, canonical_pcd: torch.Tensor
    ) -> torch.Tensor:
        """Warps and transform canonical object. Differentiable."""
        rotm = orthogonalize(self.pose_param)
        rotm = torch.bmm(rotm, self.initial_poses)
        deltas = torch.matmul(self.latent_param, components) + means
        deltas = deltas.view((self.latent_param.shape[0], -1, 3))
        new_pcd = canonical_pcd[None] + deltas
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Outputs numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(
                self.components, self.means, self.canonical_pcd
            )
            rotm = orthogonalize(self.pose_param)
            rotm = torch.bmm(rotm, self.initial_poses)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.latent_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                latent = self.latent_param[i].cpu().numpy()
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, latent, scale)
                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


class ObjectWarpingSE2Batch(ObjectWarping):
    """Object shape and planar pose warping."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        """Initializes warping parameters and optimization.

        Args:
            initial_poses: Array of initial warped object poses.
            initial_centers: Array of initial warped object centers.
            initial_latents: Array of initial warped object latent vectors.
            initial_scales: Array of initial warped object scales.
            train_latents: If False, do not optimize the shape of the object.
            train_centers: If False, do not optimize object centers.
            train_poses: If False, do not optimize object poses.
            train_scales: If False, do not optimize object scale.
        """
        # Initial poses are yaw angles here.
        n_angles = len(initial_poses)
        initial_poses_pt = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_latents is None:
            initial_latents_pt = torch.zeros(
                (n_angles, self.pca.n_components),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            initial_latents_pt = torch.tensor(
                initial_latents, dtype=torch.float32, device=self.device
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.latent_param = nn.Parameter(initial_latents_pt, requires_grad=True)
        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(initial_poses_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_latents:
            self.params.append(self.latent_param)
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor, canonical_pcd: torch.Tensor
    ) -> torch.Tensor:
        """Warps and transform canonical object. Differentiable."""
        rotm = yaw_to_rot_batch_pt(self.pose_param)
        deltas = torch.matmul(self.latent_param, components) + means
        deltas = deltas.view((self.latent_param.shape[0], -1, 3))
        new_pcd = canonical_pcd[None] + deltas
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Outputs numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(
                self.components, self.means, self.canonical_pcd
            )
            rotm = yaw_to_rot_batch_pt(self.pose_param)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.latent_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                latent = self.latent_param[i].cpu().numpy()
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, latent, scale)
                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


class ObjectSE3Batch(ObjectWarping):
    """Object pose gradient descent in SE3."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ):
        """Initializes warping parameters and optimization.

        Args:
            initial_poses: Array of initial warped object poses.
            initial_centers: Array of initial warped object centers.
            initial_latents: unused.
            initial_scales: Array of initial warped object scales.
            train_latents: unused.
            train_centers: If False, do not optimize object centers.
            train_poses: If False, do not optimize object poses.
            train_scales: If False, do not optimize object scale.
        """
        n_angles = len(initial_poses)

        # Initial rotation matrices.
        self.initial_poses = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        # This 2x3 vectors will turn into an identity rotation matrix.
        unit_ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        unit_ortho = np.repeat(unit_ortho[None], n_angles, axis=0)
        init_ortho_pt = torch.tensor(
            unit_ortho, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(init_ortho_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self,
        components: Optional[torch.Tensor],
        means: Optional[torch.Tensor],
        canonical_pcd: torch.Tensor,
    ) -> torch.Tensor:
        """Transforms canonical object. Differentiable."""
        rotm = orthogonalize(self.pose_param)
        rotm = torch.bmm(rotm, self.initial_poses)
        new_pcd = torch.repeat_interleave(
            canonical_pcd[None], len(self.pose_param), dim=0
        )
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def subsample(
        self, num_samples: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Randomly subsamples the canonical object, including its PCA projection."""
        indices = np.random.randint(0, self.canonical_pcd.shape[0], num_samples)
        return None, None, self.canonical_pcd[indices]

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Outputs numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(None, None, self.canonical_pcd)
            rotm = orthogonalize(self.pose_param)
            rotm = torch.bmm(rotm, self.initial_poses)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.center_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, None, scale)

                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


class ObjectSE2Batch(ObjectWarping):
    """Object pose gradient descent in SE2."""

    def initialize_params_and_opt(
        self,
        initial_poses: NDArray[np.float32],
        initial_centers: Optional[NDArray[np.float32]] = None,
        initial_latents: Optional[NDArray[np.float32]] = None,
        initial_scales: Optional[NDArray[np.float32]] = None,
        train_latents: bool = True,
        train_centers: bool = True,
        train_poses: bool = True,
        train_scales: bool = True,
    ) -> None:
        """Initializes warping parameters and optimization.

        Args:
            initial_poses: Array of initial warped object poses.
            initial_centers: Array of initial warped object centers.
            initial_latents: unused.
            initial_scales: Array of initial warped object scales.
            train_latents: unused.
            train_centers: If False, do not optimize object centers.
            train_poses: If False, do not optimize object poses.
            train_scales: If False, do not optimize object scale.
        """
        n_angles = len(initial_poses)
        initial_poses_pt = torch.tensor(
            initial_poses, dtype=torch.float32, device=self.device
        )

        if initial_centers is None:
            initial_centers_pt = torch.zeros(
                (n_angles, 3), dtype=torch.float32, device=self.device
            )
        else:
            initial_centers_pt = torch.tensor(
                initial_centers - self.global_means[None],
                dtype=torch.float32,
                device=self.device,
            )

        if initial_scales is None:
            initial_scales_pt = (
                torch.ones((n_angles, 3), dtype=torch.float32, device=self.device)
                * self.init_scale
            )
        else:
            initial_scales_pt = torch.tensor(
                initial_scales, dtype=torch.float32, device=self.device
            )

        self.center_param = nn.Parameter(initial_centers_pt, requires_grad=True)
        self.pose_param = nn.Parameter(initial_poses_pt, requires_grad=True)
        self.scale_param = nn.Parameter(initial_scales_pt, requires_grad=True)

        self.params = []
        if train_centers:
            self.params.append(self.center_param)
        if train_poses:
            self.params.append(self.pose_param)
        if train_scales:
            self.params.append(self.scale_param)

        self.optim = optim.Adam(self.params, lr=self.lr)

    def create_warped_transformed_pcd(
        self,
        components: Optional[torch.Tensor],
        means: Optional[torch.Tensor],
        canonical_pcd: torch.Tensor,
    ) -> torch.Tensor:
        """Warps and transforms canonical object. Differentiable."""
        rotm = yaw_to_rot_batch_pt(self.pose_param)
        new_pcd = torch.repeat_interleave(
            canonical_pcd[None], len(self.pose_param), dim=0
        )
        new_pcd = new_pcd * self.scale_param[:, None]
        new_pcd = (
            torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        )
        return new_pcd

    def subsample(
        self, num_samples: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Randomly subsamples the canonical object, including its PCA projection."""
        indices = np.random.randint(0, self.canonical_pcd.shape[0], num_samples)
        return None, None, self.canonical_pcd[indices]

    def assemble_output(
        self, cost: torch.Tensor
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:
        """Outputs numpy arrays."""
        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():
            new_pcd = self.create_warped_transformed_pcd(None, None, self.canonical_pcd)
            rotm = yaw_to_rot_batch_pt(self.pose_param)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.center_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = self.center_param[i].cpu().numpy() + self.global_means
                position = position.astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                scale = self.scale_param[i].cpu().numpy()

                obj_param = utils.ObjParam(position, quat, None, scale)

                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


def warp_to_pcd_se3(
    object_warping: Union[ObjectWarpingSE3Batch, ObjectSE3Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    """Warps to a target point cloud using SE(3) object poses.

    Args:
        object_warping: Object warping instance.
        n_angles: Number of random starting angles to sample in a batch.
        n_batches: Number of batches to run.

    Returns:
        Warped point cloud, distance to target, warping parameters.
    """
    poses = random_rots(n_angles * n_batches)

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        poses_batch = poses[batch_idx * n_angles : (batch_idx + 1) * n_angles]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            poses_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def warp_to_pcd_se3_hemisphere(
    object_warping: Union[ObjectWarpingSE3Batch, ObjectSE3Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    """Warps to a target point cloud using upright SE(3) object poses.

    Args:
        object_warping: Object warping instance.
        n_angles: Number of random starting angles to sample in a batch.
        n_batches: Number of batches to run.
        inference_kwargs: Dictionary of inference parameters.

    Returns:
        Warped point cloud, distance to target, warping parameters.
    """
    poses = random_rots_hemisphere(n_angles * n_batches)

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        poses_batch = poses[batch_idx * n_angles : (batch_idx + 1) * n_angles]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            poses_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def warp_to_pcd_se2(
    object_warping: Union[ObjectWarpingSE2Batch, ObjectSE2Batch],
    n_angles: int = 50,
    n_batches: int = 3,
    inference_kwargs={},
) -> Tuple[NDArray, float, utils.ObjParam]:
    """Warps to a target point cloud using upright SE(2) object poses.

    Args:
        object_warping: Object warping instance.
        n_angles: Number of random starting angles to sample in a batch.
        n_batches: Number of batches to run.
        inference_kwargs: Dictionary of inference parameters.

    Returns:
        Warped point cloud, distance to target, warping parameters.
    """
    start_angles = []
    for i in range(n_angles * n_batches):
        angle = i * (2 * np.pi / (n_angles * n_batches))
        start_angles.append(angle)
    start_angles = np.array(start_angles, dtype=np.float32)[:, None]

    all_costs, all_new_pcds, all_parameters = [], [], []

    for batch_idx in range(n_batches):
        start_angles_batch = start_angles[
            batch_idx * n_angles : (batch_idx + 1) * n_angles
        ]
        batch_costs, batch_new_pcds, batch_parameters = object_warping.inference(
            start_angles_batch, **inference_kwargs
        )
        all_costs += batch_costs
        all_new_pcds += batch_new_pcds
        all_parameters += batch_parameters

    best_idx = np.argmin(all_costs)
    return all_new_pcds[best_idx], all_costs[best_idx], all_parameters[best_idx]


def orthogonalize(x: torch.Tensor) -> torch.Tensor:
    """Produces an orthogonal frame from a batch of vector pairs.

    Based on https://arxiv.org/abs/2204.11371.

    Args:
        x: Batch of vector pairs of shape [B, 2, 3].

    Returns:
        Batch of 3x3 orthogonal matrices.
    """
    # u = torch.zeros([x.shape[0],3,3], dtype=torch.float32, device=x.device)
    u0 = x[:, 0] / torch.norm(x[:, 0], dim=1)[:, None]
    u1 = x[:, 1] - (torch.sum(u0 * x[:, 1], dim=1))[:, None] * u0
    u1 = u1 / torch.norm(u1, dim=1)[:, None]
    u2 = torch.cross(u0, u1, dim=1)
    return torch.stack([u0, u1, u2], dim=1)


def cost_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates the one-sided Chamfer distance between two point clouds in pytorch.

    Args:
        source: Source point cloud of shape [N, 3].
        target: Target point cloud of shape [M, 3].

    Returns:
        Scalar distance in a tensor.
    """
    diff = torch.sqrt(torch.sum(torch.square(source[:, None] - target[None, :]), dim=2))
    c = diff[list(range(len(diff))), torch.argmin(diff, dim=1)]
    return torch.mean(c)


def cost_batch_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates the one-sided Chamfer distance between two batches of point clouds in pytorch.

    Args:
        source: Source point cloud of shape [B, N, 3].
        target: Target point cloud of shape [B, M, 3].

    Returns:
        Batch of distances.
    """
    # B x N x K
    diff = torch.sqrt(
        torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3)
    )
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)


def random_rots(num: int) -> NDArray[np.float64]:
    """Samples random rotation matrices.

    Args:
        num: Number of rotations to sample.

    Batch of rotation matrices.
    """
    return Rotation.random(num=num).as_matrix().astype(np.float64)


def random_rots_hemisphere(num: int) -> NDArray[np.float32]:
    """Samples random upright rotations.

    The function samples 10 times more matrices than needed and then
        reject the ones that point downwards.

    Args:
        num: Number of rotations to sample.

    Batch of rotation matrices.
    """
    rots = random_rots(num * 10)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    out = np.einsum("bnk,kl->bnl", rots, z[:, None])[:, :, 0]
    mask = out[..., 2] >= 0
    return rots[mask][:num]


def yaw_to_rot_pt(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw angle to a rotation matrix in pytorch."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros(1, device=c.device)
    t1 = torch.ones(1, device=c.device)

    return torch.stack(
        [torch.cat([c, -s, t0]), torch.cat([s, c, t0]), torch.cat([t0, t0, t1])], dim=0
    )


def yaw_to_rot_batch_pt(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw angle to a batch of rotation matrices in pytorch."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros((yaw.shape[0], 1), device=c.device)
    t1 = torch.ones((yaw.shape[0], 1), device=c.device)

    return torch.stack(
        [
            torch.cat([c, -s, t0], dim=1),
            torch.cat([s, c, t0], dim=1),
            torch.cat([t0, t0, t1], dim=1),
        ],
        dim=1,
    )
