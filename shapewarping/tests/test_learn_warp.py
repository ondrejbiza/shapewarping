import unittest
import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from shapewarping import learn_warp


class TestCPDTransform(unittest.TestCase):
    def setUp(self):
        # Example setup
        self.source = np.array([[0, 0, 0], [1, 1, 1]])
        self.target = np.array([[1, 1, 1], [2, 2, 2]])

    def test_basic_functionality(self):
        # Test basic functionality
        result = learn_warp.cpd_transform(self.source, self.target)
        self.assertEqual(result.shape, (6,))

    def test_input_shapes(self):
        # Test for correct input shapes
        with self.assertRaises(ValueError):
            learn_warp.cpd_transform(np.array([1, 2, 3]), self.target)


class TestWarpGen(unittest.TestCase):
    def setUp(self):
        # Example setup
        self.objects = [
            np.array([[0, 0, 0], [1, 1, 1]]),
            np.array([[1, 1, 1], [2, 2, 2]]),
        ]
        self.canonical_index = 0

    def test_basic_functionality(self):
        # Test basic functionality
        warps, costs = learn_warp.warp_gen(self.canonical_index, self.objects)
        self.assertEqual(len(warps), len(self.objects) - 1)
        self.assertEqual(len(costs), len(self.objects) - 1)

    def test_invalid_index(self):
        # Test handling of invalid canonical_index
        with self.assertRaises(ValueError):
            learn_warp.warp_gen(-1, self.objects)


class TestPickCanonicalSimple(unittest.TestCase):

    def setUp(self):
        # Example setup
        self.points = [np.random.rand(10, 3), np.random.rand(10, 3), np.random.rand(10, 3)]

    def test_basic_functionality(self):
        # Test basic functionality
        index = learn_warp.pick_canonical_simple(self.points)
        self.assertIsInstance(index, int)
        self.assertTrue(0 <= index < len(self.points))

    def test_empty_list(self):
        # Test behavior with an empty list
        with self.assertRaises(ValueError):
            learn_warp.pick_canonical_simple([])

    def test_single_element_list(self):
        # Test behavior with a single element list
        with self.assertRaises(ValueError):
            learn_warp.pick_canonical_simple([])

    def test_different_sizes(self):
        # Test with different sizes of points
        points_diff_size = [np.random.rand(5, 3), np.random.rand(15, 3)]
        index = learn_warp.pick_canonical_simple(points_diff_size)
        self.assertTrue(0 <= index < len(points_diff_size))


class TestPickCanonicalWarp(unittest.TestCase):

    def setUp(self):
        # Example setup
        self.points = [np.random.rand(10, 3), np.random.rand(10, 3), np.random.rand(10, 3)]

    def test_basic_functionality(self):
        # Test basic functionality
        index = learn_warp.pick_canonical_warp(self.points)
        self.assertIsInstance(index, int)
        self.assertTrue(0 <= index < len(self.points))

    def test_invalid_input(self):
        # Test with fewer than two point clouds
        with self.assertRaises(ValueError):
            learn_warp.pick_canonical_warp([self.points[0]])


class TestLoadObjPaths(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_empty_directory(self):
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 0)

    def test_single_obj_file(self):
        subdir = self.test_dir / "obj1"
        subdir.mkdir()
        obj_file = subdir / "test.obj"
        obj_file.write_text("# Test OBJ file")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], obj_file)

    def test_multiple_obj_files_same_dir(self):
        subdir = self.test_dir / "obj1"
        subdir.mkdir()
        obj1 = subdir / "test1.obj"
        obj2 = subdir / "test2.obj"
        obj1.write_text("# Test OBJ file 1")
        obj2.write_text("# Test OBJ file 2")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0], [obj1, obj2])

    def test_preferred_name_selection(self):
        subdir = self.test_dir / "obj1"
        subdir.mkdir()
        obj1 = subdir / "test.obj"
        obj2 = subdir / "model_normalized.obj"
        obj1.write_text("# Test OBJ file 1")
        obj2.write_text("# Model normalized file")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], obj2)

    def test_multiple_subdirectories(self):
        subdir1 = self.test_dir / "obj1"
        subdir2 = self.test_dir / "obj2"
        subdir1.mkdir()
        subdir2.mkdir()
        obj1 = subdir1 / "test1.obj"
        obj2 = subdir2 / "test2.obj"
        obj1.write_text("# Test OBJ file 1")
        obj2.write_text("# Test OBJ file 2")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 2)
        self.assertIn(obj1, result)
        self.assertIn(obj2, result)

    def test_nested_directories(self):
        nested_dir = self.test_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        obj_file = nested_dir / "nested.obj"
        obj_file.write_text("# Nested OBJ file")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], obj_file)

    def test_ignores_non_obj_files(self):
        subdir = self.test_dir / "obj1"
        subdir.mkdir()
        obj_file = subdir / "test.obj"
        stl_file = subdir / "test.stl"
        txt_file = subdir / "test.txt"
        obj_file.write_text("# Test OBJ file")
        stl_file.write_text("# Test STL file")
        txt_file.write_text("# Test TXT file")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], obj_file)

    def test_case_insensitive_extensions(self):
        subdir = self.test_dir / "obj1"
        subdir.mkdir()
        obj_file = subdir / "test.OBJ"
        obj_file.write_text("# Test OBJ file")
        
        result = learn_warp.load_obj_paths(self.test_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], obj_file)

    def test_nonexistent_directory_raises_error(self):
        nonexistent_dir = Path("/nonexistent/path")
        with self.assertRaises(AssertionError):
            learn_warp.load_obj_paths(nonexistent_dir)


# class TestPCAFitTransform(unittest.TestCase):

#     def setUp(self):
#         # Example setup
#         self.distances = np.random.rand(10, 5)  # Example data

#     def test_pca_transformation(self):
#         # Test PCA transformation
#         p_components, pca = learn_warp.pca_fit_transform(self.distances, n_dimensions=3)
#         self.assertEqual(p_components.shape, (10, 3))
#         self.assertIsInstance(pca, PCA)


if __name__ == "__main__":
    unittest.main()
