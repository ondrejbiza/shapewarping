import unittest

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from shapewarping.lib import utils


class TestTransforms(unittest.TestCase):
    def test_pos_quat_to_transform_and_back(self) -> None:
        pos = np.random.normal(0, 1, 3).astype(np.float64)
        quat = (
            Rotation.from_euler("zyx", np.random.uniform(-np.pi, np.pi, 3))
            .as_quat()
            .astype(np.float64)
        )

        pos_, quat_ = utils.transform_to_pos_quat(
            utils.pos_quat_to_transform(pos, quat)
        )

        np.testing.assert_almost_equal(pos_, pos)
        # There are two unique unit quaternions per rotation matrix.
        self.assertTrue(np.allclose(quat_, quat) or np.allclose(-quat_, quat))

    def test_transform_to_pos_quat_and_back(self) -> None:
        trans = np.eye(4).astype(np.float64)
        trans[:3, :3] = Rotation.random().as_matrix()
        trans[:3, 3] = np.random.normal(0, 1, 3)

        trans_ = utils.pos_quat_to_transform(*utils.transform_to_pos_quat(trans))

        np.testing.assert_almost_equal(trans_, trans)


class TestBestFitTransform(unittest.TestCase):
    def test_identity_transform(self) -> None:
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        B = A.copy()
        expected_T = np.eye(4, dtype=np.float64)
        expected_R = np.eye(3, dtype=np.float64)
        expected_t = np.zeros((3,), dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)

    def test_translation(self) -> None:
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        B = A + np.array([2, 3, 4], dtype=np.float32)
        expected_T = np.array(
            [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=np.float64
        )
        expected_R = np.eye(3, dtype=np.float64)
        expected_t = np.array([2, 3, 4], dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)

    def test_rotation(self) -> None:
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        B = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
        expected_T = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        expected_R = expected_T[:3, :3]
        expected_t = np.zeros((3,), dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)


class TestTrimeshTransform(unittest.TestCase):
    def setUp(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def test_mesh_centering(self):
        utils.trimesh_transform(self.mesh, center=True)
        self.assertTrue(
            np.allclose(self.mesh.centroid, [0, 0, 0]), "Mesh not centered correctly"
        )

    def test_mesh_scaling(self):
        scale_factor = 2
        expected_vertices = self.mesh.vertices * scale_factor
        utils.trimesh_transform(self.mesh, center=False, scale=scale_factor)
        self.assertTrue(
            np.allclose(self.mesh.vertices, expected_vertices),
            "Mesh not scaled correctly",
        )

    def test_mesh_rotation(self):
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        utils.trimesh_transform(self.mesh, center=False, rotation=rotation_matrix)

        # Define expected rotation result
        expected_vertices = np.array(
            [[0, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
        )

        self.assertTrue(
            np.allclose(self.mesh.vertices, expected_vertices),
            "Mesh not rotated correctly",
        )

    def test_combination_transformations(self):
        scale_factor = 2
        rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        utils.trimesh_transform(
            self.mesh, center=True, scale=scale_factor, rotation=rotation_matrix
        )

        # Expected vertices after scaling, rotating, and then centering
        expected_vertices = scale_factor * np.array(
            [[0, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        expected_vertices -= np.mean(expected_vertices, axis=0)

        self.assertTrue(
            np.allclose(self.mesh.vertices, expected_vertices),
            "Combined transformations not applied correctly",
        )

    def test_no_transformation(self):
        original_vertices = np.copy(self.mesh.vertices)
        utils.trimesh_transform(self.mesh, center=False, scale=None, rotation=None)
        self.assertTrue(
            np.array_equal(self.mesh.vertices, original_vertices),
            "Mesh should not change",
        )


if __name__ == "__main__":
    unittest.main()
