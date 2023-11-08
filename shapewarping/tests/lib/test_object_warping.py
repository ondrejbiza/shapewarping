import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from shapewarping.lib import object_warping, utils


class MockPCA:
    def __init__(self, means, components):
        self.mean_ = means
        self.components_ = components
        self.n_components = len(components)


def get_mock_canon_obj_and_pca(n_latent, n_pcd):
    means = np.random.normal(0, 1, n_pcd * 3).astype(np.float32)
    components = np.random.normal(0, 1, (n_latent, n_pcd * 3)).astype(np.float32)
    pca = MockPCA(means, components)

    canonical_pcd = np.random.normal(0, 1, (n_pcd, 3)).astype(np.float32)
    pcd = np.random.normal(0, 1, (124, 3)).astype(np.float32)

    canon_obj = utils.CanonObj(
        canonical_pcd, pcd[:10], np.array([0, 1, 2], dtype=np.int32), pca
    )

    return pcd, canon_obj


class TestObjectWarping(unittest.TestCase):
    def test_init(self):
        n_latent, n_pcd = 8, 200
        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        ow = object_warping.ObjectWarping(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        np.testing.assert_almost_equal(ow.global_means, np.mean(pcd, axis=0))
        np.testing.assert_almost_equal(np.mean(ow.pcd.numpy(), axis=0), np.zeros(3))
        np.testing.assert_equal(ow.means.numpy(), canon_obj.pca.mean_)
        np.testing.assert_equal(ow.components.numpy(), canon_obj.pca.components_)

    def test_subsample(self):
        n_latent, n_pcd, n_samples = 8, 200, 20

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")
        ow = object_warping.ObjectWarping(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )

        means, components, canonical_obj_pt = ow.subsample(n_samples)

        self.assertEqual(means.numpy().shape, (n_samples * 3,))
        self.assertEqual(components.numpy().shape, (n_latent, n_samples * 3))
        self.assertEqual(canonical_obj_pt.numpy().shape, (n_samples, 3))


class TestObjectWarpingSE3Batch(unittest.TestCase):
    def test_initialize_params_and_opt_1(self):
        n_latent, n_pcd, n_batch, init_scale = 8, 200, 4, 0.2

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        pose = Rotation.random(num=n_batch).as_matrix().astype(np.float32)

        ow = object_warping.ObjectWarpingSE3Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1, init_scale=init_scale
        )
        ow.initialize_params_and_opt(pose)

        np.testing.assert_equal(ow.initial_poses.numpy(), pose)
        np.testing.assert_equal(
            ow.pose_param.data.numpy(), np.repeat(np.eye(3)[:2][None], n_batch, 0)
        )
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(
            ow.scale_param.data.numpy(),
            np.ones((n_batch, 3), dtype=np.float32) * init_scale,
        )

    def test_initialize_params_and_opt_2(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        pose = Rotation.random(num=n_batch).as_matrix().astype(np.float32)
        center = np.random.normal(0, 1, (n_batch, 3)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE3Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(pose, initial_centers=center)

        np.testing.assert_equal(ow.initial_poses.numpy(), pose)
        np.testing.assert_equal(
            ow.pose_param.data.numpy(), np.repeat(np.eye(3)[:2][None], n_batch, 0)
        )
        np.testing.assert_equal(
            ow.center_param.data.numpy(), center - ow.global_means[None]
        )
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(ow.scale_param.data.numpy(), np.ones((n_batch, 3)))

    def test_initialize_params_and_opt_3(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        pose = Rotation.random(num=n_batch).as_matrix().astype(np.float32)
        latent = np.random.normal(0, 1, (n_batch, n_latent)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE3Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(pose, initial_latents=latent)

        np.testing.assert_equal(ow.initial_poses.numpy(), pose)
        np.testing.assert_equal(
            ow.pose_param.data.numpy(), np.repeat(np.eye(3)[:2][None], n_batch, 0)
        )
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(ow.latent_param.data.numpy(), latent)
        np.testing.assert_equal(ow.scale_param.data.numpy(), np.ones((n_batch, 3)))

    def test_initialize_params_and_opt_4(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        pose = Rotation.random(num=n_batch).as_matrix().astype(np.float32)
        scale = np.random.normal(0, 1, (n_batch, 3)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE3Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(pose, initial_scales=scale)

        np.testing.assert_equal(ow.initial_poses.numpy(), pose)
        np.testing.assert_equal(
            ow.pose_param.data.numpy(), np.repeat(np.eye(3)[:2][None], n_batch, 0)
        )
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(ow.scale_param.data.numpy(), scale)


class TestObjectWarpingSE2Batch(unittest.TestCase):
    def test_initialize_params_and_opt_1(self):
        n_latent, n_pcd, n_batch, init_scale = 8, 200, 4, 0.2

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        angle = np.random.uniform(0, 2 * np.pi, n_batch).astype(np.float32)

        ow = object_warping.ObjectWarpingSE2Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1, init_scale=init_scale
        )
        ow.initialize_params_and_opt(angle)

        np.testing.assert_equal(ow.pose_param.data.numpy(), angle)
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(
            ow.scale_param.data.numpy(),
            np.ones((n_batch, 3), dtype=np.float32) * init_scale,
        )

    def test_initialize_params_and_opt_2(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        angle = np.random.uniform(0, 2 * np.pi, n_batch).astype(np.float32)
        center = np.random.normal(0, 1, (n_batch, 3)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE2Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(angle, initial_centers=center)

        np.testing.assert_equal(ow.pose_param.data.numpy(), angle)
        np.testing.assert_equal(
            ow.center_param.data.numpy(), center - ow.global_means[None]
        )
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(ow.scale_param.data.numpy(), np.ones((n_batch, 3)))

    def test_initialize_params_and_opt_3(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        angle = np.random.uniform(0, 2 * np.pi, n_batch).astype(np.float32)
        latent = np.random.normal(0, 1, (n_batch, n_latent)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE2Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(angle, initial_latents=latent)

        np.testing.assert_equal(ow.pose_param.data.numpy(), angle)
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(ow.latent_param.data.numpy(), latent)
        np.testing.assert_equal(ow.scale_param.data.numpy(), np.ones((n_batch, 3)))

    def test_initialize_params_and_opt_4(self):
        n_latent, n_pcd, n_batch = 8, 200, 4

        pcd, canon_obj = get_mock_canon_obj_and_pca(n_latent, n_pcd)
        device = torch.device("cpu")

        angle = np.random.uniform(0, 2 * np.pi, n_batch).astype(np.float32)
        scale = np.random.normal(0, 1, (n_batch, 3)).astype(np.float32)

        ow = object_warping.ObjectWarpingSE2Batch(
            canon_obj, pcd, device, **object_warping.PARAM_1
        )
        ow.initialize_params_and_opt(angle, initial_scales=scale)

        np.testing.assert_equal(ow.pose_param.data.numpy(), angle)
        np.testing.assert_equal(ow.center_param.data.numpy(), np.zeros((n_batch, 3)))
        np.testing.assert_equal(
            ow.latent_param.data.numpy(), np.zeros((n_batch, n_latent))
        )
        np.testing.assert_equal(ow.scale_param.data.numpy(), scale)


class TestObjectWarpingUtils(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)

    def test_yaw_to_rot_batch_pt(self):
        yaws_pt = torch.tensor(
            np.random.uniform(0, 2 * np.pi, (100, 1)).astype(np.float32), device="cpu"
        )

        rot_batch = object_warping.yaw_to_rot_batch_pt(yaws_pt).numpy()
        rots = []
        for yaw_pt in yaws_pt:
            rots.append(object_warping.yaw_to_rot_pt(yaw_pt).numpy())
        rots = np.array(rots)
        np.testing.assert_almost_equal(rot_batch, rots)

    def test_cost_batch_pt(self):
        source = torch.tensor(
            np.random.normal(0, 1, (10, 3217, 3)).astype(np.float32), device="cpu"
        )
        target = torch.tensor(
            np.random.normal(0, 1, (10, 1234, 3)).astype(np.float32), device="cpu"
        )

        cost_batch = object_warping.cost_batch_pt(source, target).numpy()
        costs = []
        for ss, tt in zip(source, target):
            costs.append(object_warping.cost_pt(ss, tt).numpy())
        costs = np.array(costs)
        np.testing.assert_almost_equal(cost_batch, costs)

    def test_orthogonalize_unit_rotation(self):
        x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

        y = object_warping.orthogonalize(x)[0].cpu().numpy()
        ref = np.eye(3)

        np.testing.assert_equal(y, ref)


if __name__ == "__main__":
    unittest.main()
