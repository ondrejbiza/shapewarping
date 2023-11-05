import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from shapewarping.lib import utils

class TestTransforms(unittest.TestCase):

    def test_pos_quat_to_transform_and_back(self) -> None:
        
        pos = np.random.normal(0, 1, 3).astype(np.float64)
        quat = Rotation.from_euler("zyx", np.random.uniform(-np.pi, np.pi, 3)).as_quat().astype(np.float64)

        pos_, quat_ = utils.transform_to_pos_quat(utils.pos_quat_to_transform(pos, quat))
        
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
        A = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
        B = A.copy()
        expected_T = np.eye(4, dtype=np.float64)
        expected_R = np.eye(3, dtype=np.float64)
        expected_t = np.zeros((3,), dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)

    def test_translation(self) -> None:
        A = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
        B = A + np.array([2, 3, 4], dtype=np.float32)
        expected_T = np.array([[1, 0, 0, 2],
                               [0, 1, 0, 3],
                               [0, 0, 1, 4],
                               [0, 0, 0, 1]], dtype=np.float64)
        expected_R = np.eye(3, dtype=np.float64)
        expected_t = np.array([2, 3, 4], dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)

    def test_rotation(self) -> None:
        A = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float32)
        B = np.array([[ 0,  0,  0],
                      [ 0,  1,  0],
                      [-1,  0,  0],
                      [ 0,  0,  1]], dtype=np.float32)
        expected_T = np.array([[ 0, -1,  0,  0],
                               [ 1,  0,  0,  0],
                               [ 0,  0,  1,  0],
                               [ 0,  0,  0,  1]], dtype=np.float64)
        expected_R = expected_T[:3, :3]
        expected_t = np.zeros((3,), dtype=np.float64)

        result_T, result_R, result_t = utils.best_fit_transform(A, B)
        np.testing.assert_array_almost_equal(result_T, expected_T, decimal=5)
        np.testing.assert_array_almost_equal(result_R, expected_R, decimal=5)
        np.testing.assert_array_almost_equal(result_t, expected_t, decimal=5)
