import unittest
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import pybullet as pb

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


class TestFarthestPointSample(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_same_number_of_points(self):
        point = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        npoint = 4
        expected_point = np.array([[0, 1, 0],
                                   [1, 0, 0],
                                   [0, 0, 1],
                                   [0, 0, 0]])
        expected_indices = np.array([2, 1, 3, 0], dtype=np.int32)
        result_point, result_indices = utils.farthest_point_sample(point, npoint)
        np.testing.assert_array_almost_equal(result_point, expected_point, decimal=5)
        np.testing.assert_array_equal(result_indices, expected_indices)

    def test_half_number_of_points(self):
        point = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        npoint = 2
        expected_point = np.array([[0, 1, 0],
                                   [1, 0, 0]])
        expected_indices = np.array([2, 1], dtype=np.int32)
        result_point, result_indices = utils.farthest_point_sample(point, npoint)
        np.testing.assert_array_almost_equal(result_point, expected_point, decimal=5)
        np.testing.assert_array_equal(result_indices, expected_indices)

    def test_more_points_than_input(self):
        point = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        npoint = 6
        with self.assertRaises(ValueError):
            utils.farthest_point_sample(point, npoint)

    def test_single_point(self):
        point = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        npoint = 1
        expected_indices = np.array([2], dtype=np.int32)
        _, result_indices = utils.farthest_point_sample(point, npoint)
        np.testing.assert_array_equal(result_indices, expected_indices)


class TestRotationDistance(unittest.TestCase):

    def test_identity(self):
        A = np.eye(3)
        B = np.eye(3)
        expected = 0.0
        self.assertAlmostEqual(utils.rotation_distance(A, B), expected, places=5)

    def test_90_degree_rotation(self):
        A = np.eye(3)
        B = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]])
        expected = np.pi / 2
        self.assertAlmostEqual(utils.rotation_distance(A, B), expected, places=5)

    def test_180_degree_rotation(self):
        A = np.eye(3)
        B = np.array([[-1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  1]])
        expected = np.pi
        self.assertAlmostEqual(utils.rotation_distance(A, B), expected, places=5)

    def test_numerical_stability(self):
        A = np.eye(3)
        B = np.array([[ 1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0, -1]])
        expected = np.pi
        self.assertAlmostEqual(utils.rotation_distance(A, B), expected, places=5)


class TestPoseDistance(unittest.TestCase):

    def test_same_pose(self):
        trans1 = np.eye(4)
        trans2 = np.eye(4)
        expected = 0.0
        result = utils.pose_distance(trans1, trans2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_translation_only(self):
        trans1 = np.eye(4)
        trans2 = np.eye(4)
        trans2[:3, 3] = np.array([2, 0, 0])
        expected = 2.0
        result = utils.pose_distance(trans1, trans2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_rotation_only(self):
        trans1 = np.eye(4)
        trans2 = np.eye(4)
        trans2[:3, :3] = np.array([[0, -1, 0],
                                   [1,  0, 0],
                                   [0,  0, 1]])
        expected = np.pi / 2
        result = utils.pose_distance(trans1, trans2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_combined_translation_and_rotation(self):
        trans1 = np.eye(4)
        trans2 = np.eye(4)
        trans2[:3, 3] = np.array([2, 0, 0])
        trans2[:3, :3] = np.array([[0, -1, 0],
                                   [1,  0, 0],
                                   [0,  0, 1]])
        expected = 2.0 + np.pi / 2
        result = utils.pose_distance(trans1, trans2)
        self.assertAlmostEqual(result, expected, places=5)


class TestWiggle(unittest.TestCase):

    def setUp(self):
        self.sim_id = pb.connect(pb.DIRECT)
        pb.resetSimulation()
        test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        self.source_obj = pb.loadURDF(os.path.join(test_data_path, "box_0.urdf"))
        self.target_obj = pb.loadURDF(os.path.join(test_data_path, "box_1.urdf"))

    def tearDown(self):
        pb.disconnect(self.sim_id)

    def test_no_collision_initially(self):
        utils.pb_set_pose(self.source_obj, np.array([1, 1, 1]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)
        utils.pb_set_pose(self.target_obj, np.array([3, 3, 3]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)

        pos, quat = utils.pb_get_pose(self.source_obj, sim_id=self.sim_id)
        result_pos, result_quat = utils.wiggle(self.source_obj, self.target_obj, sim_id=self.sim_id)

        self.assertTrue(np.allclose(result_pos, pos))
        self.assertTrue(np.allclose(result_quat, quat))

    def test_collision(self):
        utils.pb_set_pose(self.source_obj, np.array([1, 1, 1]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)
        utils.pb_set_pose(self.target_obj, np.array([1.1, 1.1, 1.1]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)

        result_pos, result_quat = utils.wiggle(self.source_obj, self.target_obj, sim_id=self.sim_id)

        pb.performCollisionDetection()
        in_collision = utils.pb_body_collision(self.source_obj, self.target_obj, sim_id=self.sim_id)
        self.assertFalse(in_collision)

    def test_no_solution_found(self):
        utils.pb_set_pose(self.source_obj, np.array([1, 1, 1]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)
        utils.pb_set_pose(self.target_obj, np.array([1.1, 1.1, 1.1]), np.array([0, 0, 0, 1]), sim_id=self.sim_id)

        pos, quat = utils.pb_get_pose(self.source_obj, sim_id=self.sim_id)
        result_pos, result_quat = utils.wiggle(self.source_obj, self.target_obj, max_tries=1, sd=0.001, sim_id=self.sim_id)

        self.assertTrue(np.allclose(result_pos, pos))
        self.assertTrue(np.allclose(result_quat, quat))


if __name__ == "__main__":
    unittest.main()
