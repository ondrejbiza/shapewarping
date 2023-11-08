from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import pickle

import trimesh
import numpy as np
import numpy.typing as npt
import torch
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation

NPF32 = npt.NDArray[np.float32]
NPF64 = npt.NDArray[np.float64]
NPI32 = npt.NDArray[np.int32]


@dataclass
class ObjParam:
    """Object shape and pose parameters.
    """
    position: NPF64 = np.array([0., 0., 0.])
    quat: NPF64 = np.array([0., 0., 0., 1.])
    latent: Optional[NPF32] = None
    scale: NPF32 = np.array([1., 1., 1.], dtype=np.float32)

    def get_transform(self) -> NPF64:
        return pos_quat_to_transform(self.position, self.quat)


@dataclass
class CanonObj:
    """Canonical object with shape warping.
    """
    canonical_pcd: NPF32
    mesh_vertices: NPF32
    mesh_faces: NPI32
    pca: Optional[PCA] = None

    def __post_init__(self):
        if self.pca is not None:
            self.n_components = self.pca.n_components

    def to_pcd(self, obj_param: ObjParam) -> NPF32:
        if self.pca is not None and obj_param.latent is not None:
            pcd = self.canonical_pcd + self.pca.inverse_transform(obj_param.latent).reshape(-1, 3)
        else:
            if self.pca is not None:
                print("WARNING: Skipping warping because we do not have a latent vector. We however have PCA.")
            pcd = np.copy(self.canonical_pcd)
        return pcd * obj_param.scale[None]

    def to_transformed_pcd(self, obj_param: ObjParam) -> NPF32:
        pcd = self.to_pcd(obj_param)
        trans = pos_quat_to_transform(obj_param.position, obj_param.quat)
        return transform_pcd(pcd, trans)

    def to_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        pcd = self.to_pcd(obj_param)
        # The vertices are assumed to be at the start of the canonical_pcd.
        vertices = pcd[:len(self.mesh_vertices)]
        return trimesh.Trimesh(vertices, self.mesh_faces)

    def to_transformed_mesh(self, obj_param: ObjParam) -> trimesh.Trimesh:
        pcd = self.to_transformed_pcd(obj_param)
        # The vertices are assumed to be at the start of the canonical_pcd.
        vertices = pcd[:len(self.mesh_vertices)]
        return trimesh.Trimesh(vertices, self.mesh_faces)

    @staticmethod
    def from_pickle(load_path: str) -> "CanonObj":
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        pcd = data["canonical_obj"]
        pca = None
        if "pca" in data:
            pca = data["pca"]
        mesh_vertices = data["canonical_mesh_points"]
        mesh_faces = data["canonical_mesh_faces"]
        return CanonObj(pcd, mesh_vertices, mesh_faces, pca)


@dataclass
class PickDemoContactPoints:
    """Pick demonstrating with contact points between object and gripper.
    """
    gripper_indices: NPI32
    target_indices: NPI32

    def check_consistent(self):
        assert len(self.gripper_indices) == len(self.target_indices)


@dataclass
class PlaceDemoVirtualPoints:
    """Place demonstration with virtual points.
    """
    knns: NPI32
    deltas: NPF32
    target_indices: NPI32

    def check_consistent(self):
        assert len(self.knns) == len(self.deltas) == len(self.target_indices)


def quat_to_rotm(quat: NPF64) -> NPF64:
    return Rotation.from_quat(quat).as_matrix()


def rotm_to_quat(rotm: NPF64) -> NPF64:
    return Rotation.from_matrix(rotm).as_quat()


def pos_quat_to_transform(
        pos: Union[Tuple[float, float, float], NPF64],
        quat: Union[Tuple[float, float, float, float], NPF64]
    ) -> NPF64:
    trans = np.eye(4).astype(np.float64)
    trans[:3, 3] = pos
    trans[:3, :3] = quat_to_rotm(np.array(quat))
    return trans


def transform_to_pos_quat(trans: NPF64) -> Tuple[NPF64, NPF64]:
    pos = trans[:3, 3]
    quat = rotm_to_quat(trans[:3, :3])
    # Just making sure.
    return pos.astype(np.float64), quat.astype(np.float64)


def transform_to_pos_rot(trans: NPF64) -> Tuple[NPF64, NPF64]:
    pos = trans[:3, 3]
    rot = trans[:3, :3]
    # Just making sure.
    return pos.astype(np.float64), rot.astype(np.float64)


def transform_pcd(pcd: NPF32, trans: NPF64, is_position: bool=True) -> NPF32:
    n = pcd.shape[0]
    cloud = pcd.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0: 3, :].T
    return cloud


def best_fit_transform(A: NPF32, B: NPF32) -> Tuple[NPF64, NPF64, NPF64]:
    '''
    https://github.com/ClayFlannigan/icp/blob/master/icp.py
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1).astype(np.float64)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R.astype(np.float64), t.astype(np.float64)


def trimesh_load_object(obj_path: str) -> trimesh.Trimesh:
    """Loads a mesh using trimesh."""
    return trimesh.load(obj_path)


def trimesh_transform(mesh: trimesh.Trimesh, center: bool=True, 
                      scale: Optional[float]=None, rotation: Optional[npt.NDArray]=None) -> None:
    """Optionally centers, scales and rotates a mesh.
    
    Args:
        mesh: Trimesh mesh.
        center: Center the mesh.
        scale: Scale the mesh by a scalar in all axes.
        rotation: 3D rotation matrix.
    """

    # Automatically center. Also possibly rotate and scale.
    translation_matrix = np.eye(4)
    scaling_matrix = np.eye(4)
    rotation_matrix = np.eye(4)

    if center:
        t = mesh.centroid
        translation_matrix[:3, 3] = -t

    if scale is not None:
        scaling_matrix[0, 0] *= scale
        scaling_matrix[1, 1] *= scale
        scaling_matrix[2, 2] *= scale

    if rotation is not None:
        rotation_matrix[:3, :3] = rotation

    transform = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))
    mesh.apply_transform(transform)


def trimesh_create_verts_surface(mesh: trimesh.Trimesh, num_surface_samples: Optional[int]=1500) -> npt.NDArray[np.float32]:
    """Samples points on the surface of a mesh.

    Note that trimesh performs rejection sampling and does not return
        exactly the specific number of points.

    Args:
        mesh: Trimesh mesh.
        num_surface_samples: Number of samples on the mesh surface.
    
    Returns:
        Array of sampled points.
    """
    surf_points, _ = trimesh.sample.sample_surface_even(
        mesh, num_surface_samples
    )
    return np.array(surf_points, dtype=np.float32)


def trimesh_get_vertices_and_faces(mesh: trimesh.Trimesh) -> Tuple[npt.NDArray[np.float32],npt. NDArray[np.int32]]:
    """Returns vertices and faces of a mesh."""
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def scale_points_circle(points: List[npt.NDArray[np.float32]], base_scale: float=1.) -> List[npt.NDArray[np.float32]]:
    """Scales points to fit inside of a circle.
    
    Args:
        points: Array of points.
        base_scale: Radius of the circle.
    
    Returns:
        Scaled points.
    """
    points_cat = np.concatenate(points)
    assert len(points_cat.shape) == 2

    length = np.sqrt(np.sum(np.square(points_cat), axis=1))
    max_length = np.max(length, axis=0)

    new_points = []
    for p in points:
        new_points.append(base_scale * (p / max_length))

    return new_points


def chamfer_distance_batch(source: npt.NDArray, target: npt.NDArray) -> float:
    """Computes the asymmetric chamfer distance between a pair of point clouds.
    
    Chamfer distance finds the closest point in target for each point in source.
        It then averages these distances.

    Args:
        source: Source point cloud.
        target: Target point cloud.
    
    Returns:
        Chamfer distance scalar.
    """
    idx = np.sum(np.abs(source[None, :] - target[:, None]), axis=2).argmin(axis=0)
    return np.mean(np.linalg.norm(source - target[idx], axis=1))


def chamfer_distance_batch_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the asymmetric chamfer distance between a pair of point clouds.
    
    Chamfer distance finds the closest point in target for each point in source.
        It then averages these distances.

    Args:
        source: Source point cloud.
        target: Target point cloud.
    
    Returns:
        Chamfer distance scalar.
    """
    # for each vertex in source, find the closest vertex in target
    # we don't need to propagate the gradient here
    source_d, target_d = source.detach(), target.detach()
    indices = (source_d[:, :, None] - target_d[:, None, :]).square().sum(dim=3).argmin(dim=2)

    # go from [B, indices_in_target, 3] to [B, indices_in_source, 3] using target[batch_indices, indices]
    batch_indices = torch.arange(0, indices.size(0), device=indices.device)[:, None].repeat(1, indices.size(1))
    c = torch.sqrt(torch.sum(torch.square(source - target[batch_indices, indices]), dim=2))
    return torch.mean(c, dim=1)

    # simple version, about 2x slower
    # bigtensor = source[:, :, None] - target[:, None, :]
    # diff = torch.sqrt(torch.sum(torch.square(bigtensor), dim=3))
    # c = torch.min(diff, dim=2)[0]
    # return torch.mean(c, dim=1)
