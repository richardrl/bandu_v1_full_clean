from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import glob
import os
import shutil
import sys
import itertools

import numpy as np
import pybullet as p
import torch
from scipy.spatial.transform import Rotation as R

from imports.QuaterNet.quaternion import qrot
import open3d
import copy
from functools import singledispatch

class RigidbodyTransform:
    def __init__(self, rotation_matrix, translation):
        self.rotation = rotation_matrix
        self.translation = translation

        X = np.zeros((4, 4))
        X[:3, :3] = rotation_matrix
        X[:3, 3] = translation
        X[3, 3] = 1
        self.homogeneous_matrix = X

    # def multiply(self, points):
    #     """
    #     Right multiplies points by the rigid transform
    #     :param points: 3 x N
    #     :return: transformed_points: 3 x N
    #     """
    #     assert points.shape[0] == 3
    #     # homogenize points
    #     points = np.concatenate((points, np.ones((1, points.shape[-1]))), axis=0)
    #
    #     return (self.homogeneous_matrix @ points)[:3, :]

    @singledispatch
    def multiply(self, arg):
        print("ln42 multiply type")
        print(type(arg))
        raise NotImplementedError

    @multiply.register
    def _(self, arg: np.ndarray):
        """
        Right multiplies points by the rigid transform
        :param points: 3 x N
        :return: transformed_points: 3 x N
        """
        assert arg.shape[0] == 3
        # homogenize points
        points = np.concatenate((arg, np.ones((1, arg.shape[-1]))), axis=0)

        return (self.homogeneous_matrix @ points)[:3, :]

    def inv(self):
        new = copy.deepcopy(self)
        new.homogeneous_matrix = np.linalg.inv(self.homogeneous_matrix)

        new.rotation = new.homogeneous_matrix[:3, :3]
        new.translation = new.homogeneous_matrix[:3, 3]

        return new


@RigidbodyTransform.multiply.register(RigidbodyTransform)
def _(self, arg: RigidbodyTransform):
    """
    Composes rigid body transforms
    :param arg:
    :return:
    """
    new_homo_mat = self.homogeneous_matrix @ arg.homogeneous_matrix
    return RigidbodyTransform(new_homo_mat[:3, :3], new_homo_mat[:3, 3])


def get_relative_rotation(rotated_pointcloud,
                          binary_logits,
                          sigmoid_threshold=.5,
                          dir="surface_to_upright"):
    """

    :param rotated_pointcloud: num_points x 3
    :param binary_logits: num_points
    :param dir: whether we find the rotation from the surface normal to the -z, or the rotation from -z to the surface normal
    :param com: center of mass, used to determine oriented normal
    :return:
    """
    assert len(rotated_pointcloud.shape) == 2
    assert len(binary_logits.shape) == 1
    assert dir in ["surface_to_upright", "upright_to_surface"]
    surface_points = rotated_pointcloud[torch.sigmoid(binary_logits) < sigmoid_threshold]
    surface_pcd = open3d.geometry.PointCloud()
    surface_pcd.points = open3d.utility.Vector3dVector(surface_points.cpu().data.numpy())
    plane_model, plane_idxs = surface_pcd.segment_plane(.007, 15, 1000)
    plane_normal = np.array(plane_model)[:3]
    a, b, c, d = plane_model

    # orient the plane normal away from the center of mass
    # normal should have the same sign as the vector from the center of mass to the plane origin
    oriented_normal = np.sign(-d) * plane_normal

    if dir == "surface_to_upright":
        return bandu_util.get_rotation_matrix_between_vecs([0, 0, -1], oriented_normal)
    else:
        return bandu_util.get_rotation_matrix_between_vecs(oriented_normal, [0, 0, -1])


def ang_in_mpi_ppi(angle):
    """
    Convert the angle to the range [-pi, pi).

    Args:
        angle (float): angle in radians.

    Returns:
        float: equivalent angle in [-pi, pi).
    """

    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def clamp(n, minn, maxn):
    """
    Clamp the input value to be in [minn, maxn].

    Args:
        n (float or int): input value.
        minn (float or int): minimum value.
        maxn (float or int): maximum value.

    Returns:
        float or int: clamped value.
    """
    return max(min(maxn, n), minn)


def quat2rot(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: rotation matrix (shape: :math:`[3, 3]`).

    """
    r = R.from_quat(quat)
    return r.as_dcm()


def quat2euler(quat, axes='xyz'):
    """
    Convert quaternion to euler angles.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).
        axes (str): Specifies sequence of axes for predicted_rotations_quat.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic predicted_rotations_quat (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic predicted_rotations_quat (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: euler angles (shape: :math:`[3,]`).
    """
    r = R.from_quat(quat)
    return r.as_euler(axes)


def quat2rotvec(quat):
    """
    Convert quaternion to rotation vector.

    Arguments:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: rotation vector (shape: :math:`[3,]`).
    """
    r = R.from_quat(quat)
    return r.as_rotvec()


def quat_inverse(quat):
    """
    Return the quaternion inverse.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: inverse quaternion (shape: :math:`[4,]`).
    """
    r = R.from_quat(quat)
    return r.inv().as_quat()


def quat_multiply(quat1, quat2):
    """
    Quaternion mulitplication.

    Args:
        quat1 (list or np.ndarray): first quaternion [x,y,z,w]
            (shape: :math:`[4,]`).
        quat2 (list or np.ndarray): second quaternion [x,y,z,w]
            (shape: :math:`[4,]`).

    Returns:
        np.ndarray: quat1 * quat2 (shape: :math:`[4,]`).
    """
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    r = r1 * r2
    return r.as_quat()


def rotvec2rot(vec):
    """
    A rotation vector is a 3 dimensional vector which is
    co-directional to the canonical_axis of rotation and whose
    norm gives the angle of rotation (in radians).

    Args:
        vec (list or np.ndarray): a rotational vector. Its norm
            represents the angle of rotation.

    Returns:
        np.ndarray: rotation matrix (shape: :math:`[3, 3]`).
    """
    r = R.from_rotvec(vec)
    return r.as_dcm()


def rotvec2quat(vec):
    """
    A rotation vector is a 3 dimensional vector which is
    co-directional to the canonical_axis of rotation and whose
    norm gives the angle of rotation (in radians).

    Args:
        vec (list or np.ndarray): a rotational vector. Its norm
            represents the angle of rotation.

    Returns:
        np.ndarray: quaternion [x,y,z,w] (shape: :math:`[4,]`).
    """
    r = R.from_rotvec(vec)
    return r.as_quat()


def rotvec2euler(vec, axes='xyz'):
    """
    A rotation vector is a 3 dimensional vector which is
    co-directional to the canonical_axis of rotation and whose
    norm gives the angle of rotation (in radians).

    Args:
        vec (list or np.ndarray): a rotational vector. Its norm
            represents the angle of rotation.
        axes (str): Specifies sequence of axes for predicted_rotations_quat.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic predicted_rotations_quat (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic predicted_rotations_quat (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: euler angles (shape: :math:`[3,]`).
    """
    r = R.from_rotvec(vec)
    return r.as_euler(axes)


def euler2rot(euler, axes='xyz'):
    """
    Convert euler angles to rotation matrix.

    Args:
        euler (list or np.ndarray): euler angles (shape: :math:`[3,]`).
        axes (str): Specifies sequence of axes for predicted_rotations_quat.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic predicted_rotations_quat (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic predicted_rotations_quat (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: rotation matrix (shape: :math:`[3, 3]`).
    """
    r = R.from_euler(axes, euler)
    return r.as_dcm()


def euler2quat(euler, axes='xyz'):
    """
    Convert euler angles to quaternion.

    Args:
        euler (list or np.ndarray): euler angles (shape: :math:`[3,]`).
        axes (str): Specifies sequence of axes for predicted_rotations_quat.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic predicted_rotations_quat (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic predicted_rotations_quat (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: quaternion [x,y,z,w] (shape: :math:`[4,]`).
    """
    r = R.from_euler(axes, euler)
    return r.as_quat()


def rot2quat(rot):
    """
    Convert rotation matrix to quaternion.

    Args:
        rot (np.ndarray): rotation matrix (shape: :math:`[3, 3]`).

    Returns:
        np.ndarray: quaternion [x,y,z,w] (shape: :math:`[4,]`).
    """
    r = R.from_dcm(rot)
    return r.as_quat()


def rot2euler(rot, axes='xyz'):
    """
    Convert rotation matrix to euler angles.

    Args:
        rot (np.ndarray): rotation matrix (shape: :math:`[3, 3]`).
        axes (str): Specifies sequence of axes for predicted_rotations_quat.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic predicted_rotations_quat (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic predicted_rotations_quat (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: euler angles (shape: :math:`[3,]`).
    """
    r = R.from_dcm(rot)
    return r.as_euler(axes)


def print_red(skk):
    """
    print the text in red color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[91m {}\033[00m".format(skk))


def print_green(skk):
    """
    print the text in green color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[92m {}\033[00m".format(skk))


def print_yellow(skk):
    """
    print the text in yellow color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[93m {}\033[00m".format(skk))


def print_blue(skk):
    """
    print the text in blue color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[94m {}\033[00m".format(skk))


def print_purple(skk):
    """
    print the text in purple color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[95m {}\033[00m".format(skk))


def print_cyan(skk):
    """
    print the text in cyan color.

    Args:
        skk (str): text to be printed.
    """
    bandu_logger.debug("\033[96m {}\033[00m".format(skk))


def create_folder(path, delete=True):
    """
    Create a new folder.

    Args:
        path (str): path of the folder.
        delete (bool): if delete=True, then if the path already
            exists, the folder will be deleted and recreated.
    """
    if delete and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def list_class_names(dir_path):
    """
    Return the mapping of class names in all files
    in dir_path to their file path.

    Args:
        dir_path (str): absolute path of the folder.

    Returns:
        dict: mapping from the class names in all python files in the
        folder to their file path.

    """

    py_files = glob.glob(os.path.join(dir_path, "*.py"))
    py_files = [f for f in py_files if os.path.isfile(f) and
                not f.endswith('__init__.py')]
    cls_name_to_path = dict()
    for py_file in py_files:
        with open(py_file) as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path


def load_class_from_path(cls_name, path):
    """
    Load a class from the file path.

    Args:
        cls_name (str): class name.
        path (str): python file path.

    Returns:
        Python Class: return the class A which is named as cls_name.
        You can call A() to create an instance of this class using
        the return value.

    """
    mod_name = 'MOD%s' % cls_name
    if sys.version_info.major == 2:
        import imp
        mod = imp.load_source(mod_name, path)
        return getattr(mod, cls_name)
    elif sys.version_info.major == 3:
        if sys.version_info.minor < 5:
            from importlib.machinery import SourceFileLoader

            mod = SourceFileLoader(mod_name, path).load_module()
            return getattr(mod, cls_name)
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, cls_name)
    else:
        raise NotImplementedError


def linear_interpolate_path(start_pos, delta_xyz, interval):
    """
    Linear interpolation in a path.

    Args:
        start_pos (list or np.ndarray): start position
            ([x, y, z], shape: :math:`[3]`).
        delta_xyz (list or np.ndarray): movement in x, y, z
            directions (shape: :math:`[3,]`).
        interval (float): interpolation interval along delta_xyz.
            Interpolate a point every `interval` distance
            between the two end points.

    Returns:
        np.ndarray: waypoints along the path (shape: :math:`[N, 3]`).

    """
    start_pos = np.array(start_pos).flatten()
    delta_xyz = np.array(delta_xyz).flatten()
    path_len = np.linalg.norm(delta_xyz)
    num_pts = int(np.ceil(path_len / float(interval)))
    if num_pts <= 1:
        num_pts = 2
    waypoints_sp = np.linspace(0, path_len, num_pts).reshape(-1, 1)
    waypoints = start_pos + waypoints_sp / float(path_len) * delta_xyz
    return waypoints


def to_rot_mat(ori):
    """
    Convert orientation in any form (rotation matrix,
    quaternion, or euler angles) to rotation matrix.

    Args:
        ori (list or np.ndarray): orientation in any following form:
            rotation matrix (shape: :math:`[3, 3]`)
            quaternion (shape: :math:`[4]`)
            euler angles (shape: :math:`[3]`).

    Returns:
        np.ndarray: orientation matrix (shape: :math:`[3, 3]`).
    """

    ori = np.array(ori)
    if ori.size == 3:
        # [roll, pitch, yaw]
        ori = euler2rot(ori)
    elif ori.size == 4:
        ori = quat2rot(ori)
    elif ori.shape != (3, 3):
        raise ValueError('Orientation should be rotation matrix, '
                         'euler angles or quaternion')
    return ori


def to_euler_angles(ori):
    """
    Convert orientation in any form (rotation matrix,
    quaternion, or euler angles) to euler angles (roll, pitch, yaw).

    Args:
        ori (list or np.ndarray): orientation in any following form:
            rotation matrix (shape: :math:`[3, 3]`)
            quaternion (shape: :math:`[4]`)
            euler angles (shape: :math:`[3]`).

    Returns:
        np.ndarray: euler angles [roll, pitch, yaw] (shape: :math:`[3,]`).

    """
    ori = np.array(ori)
    if ori.size == 4:
        ori = quat2euler(ori)
    elif ori.shape == (3, 3):
        ori = rot2euler(ori)
    elif ori.size != 3:
        raise ValueError('Orientation should be rotation matrix, '
                         'euler angles or quaternion')
    return ori


def to_quat(ori):
    """
    Convert orientation in any form (rotation matrix,
    quaternion, or euler angles) to quaternion.

    Args:
        ori (list or np.ndarray): orientation in any following form:
            rotation matrix (shape: :math:`[3, 3]`)
            quaternion (shape: :math:`[4]`)
            euler angles (shape: :math:`[3]`).

    Returns:
        np.ndarray: quaternion [x, y, z, w](shape: :math:`[4, ]`).
    """
    ori = np.array(ori)
    if ori.size == 3:
        # [roll, pitch, yaw]
        ori = euler2quat(ori)
    elif ori.shape == (3, 3):
        ori = rot2quat(ori)
    elif ori.size != 4:
        raise ValueError('Orientation should be rotation matrix, '
                         'euler angles or quaternion')
    return ori


def homogenize_matrix(rot_quat, trans):
    """

    :param rot_quat:
    :param trans:
    :return: 4x4 transformation matrix
    """
    # if matrix.shape != torch.Size([3, 3]):
    #     bandu_logger.debug("Matrix is already 4x4, cannot homogenize.")
    #     return matrix
    if type(rot_quat) == tuple:
        rot_quat = np.array(rot_quat)
    if type(trans) == tuple:
        trans = np.array(trans)

    assert rot_quat.shape == (4,), rot_quat.shape
    assert trans.shape == (3,), trans.shape

    temp_mat = np.zeros((4, 4))
    temp_mat[:3, :3] = R.from_quat(rot_quat).as_matrix()
    temp_mat[3, 3] = 1
    temp_mat[:3, 3] = trans
    return temp_mat


def inverse_homogenize_matrix(matrix):
    return matrix[0:3, 3], R.from_matrix(matrix[0:3, 0:3]).as_quat()


def quat_inverse_torch(quat):
    """
    OMG! THIS FUNCTION WAS OPERATING ON QUAT IN PLACE!!!
    :param quat: (X,Y,Z,W) pybullet standard
    :return:
    """
    # torch.set_printoptions(precision=10)
    # assert torch.isclose(quat.norm(), torch.Tensor(1), rtol=1e-04, atol=1e-04), (quat.norm(), quat.shape)
    if isinstance(quat, np.ndarray):
        out_quat = quat.copy()
    else:
        out_quat = quat.clone()
    assert quat.shape[-1] == 4
    if len(quat.shape) == 1:
        out_quat[:3] = out_quat[:3] * -1
    elif len(quat.shape) == 2:
        out_quat[:, :3] = out_quat[:, :3] * -1
    elif len(quat.shape) == 3:
        out_quat[:, :, :3] = out_quat[:, :, :3] * -1
    else:
        bandu_logger.debug("quat.shape")
        bandu_logger.debug(quat.shape)
        raise NotImplementedError
    return out_quat


def body_frame_transform(start_point_cloud,
                         start_quat,
                         start_trans,
                         relative_body_quat,
                         relative_body_trans,
                         device=None):
    # canonical_frame_point_cloud = qrot(torch_quat_inverse(start_quat), (start_point_cloud - start_trans)).to(device)
    # -> [batch_size, num_points, 3]
    points_relative_body = qrot(relative_body_quat, start_point_cloud) + relative_body_trans
    # Points after applying relative body, and body world transforms
    points_body_world = qrot(start_quat, points_relative_body) + start_trans
    return points_body_world


def torch_quat2mat(quat):
    # from DCP code
    # Quaternion to rotation matrix
    if len(quat.shape) == 1:
        quat = quat.unsqueeze(0)
    if len(quat.shape) == 2:
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
        return rotMat
    elif len(quat.shape) == 3:
        # nO: num_objects
        nO = quat.size(1)

        # select one index out of [nB, nO, 4] -> [nB, nO] for each scalar component
        x, y, z, w = quat[:, :, 0], quat[:, :, 1], quat[:, :, 2], quat[:, :, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)

        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                              2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                              2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=2).reshape(B, nO, 3, 3)
        return rotMat
    else:
        raise NotImplementedError


# def get_rotation_matrices_between_vec_arrays(target_vec_arr, start_vec_arr):
#     rotmats = []
#     for obj_idx in range(surface_normals.shape[1]):
#         rotmat = transform_util.get_rotation_matrix_between_vecs([0, 0, -1], surface_normals[sample_idx][obj_idx][0].cpu().data.numpy())
#         rotmats.append(rotmat)
#     return rotmats


def get_rotation_matrix_between_vecs(target_vec, start_vec):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # rotation with theta = the cosine angle between the two vectors
    """

    :param target_vec:
    :param start_vec:
    :return: 3x3 rotation matrix representing relative rotation from start_vec to target_vec
    """
    # the formula fails when the vectors are collinear

    target_vec = np.array(target_vec)
    start_vec = np.array(start_vec)
    assert len(target_vec.shape) == 1 or len(target_vec.shape) == 2
    assert target_vec.shape == start_vec.shape, (target_vec.shape, start_vec.shape)
    target_vec = target_vec / np.linalg.norm(target_vec, axis=-1)
    start_vec = start_vec/np.linalg.norm(start_vec, axis=-1)

    # the formula doesn't work in this case...
    # TODO: make this an actual rotation instead of a flip...
    if np.all(np.isclose(start_vec, -1 * target_vec, atol=1.e-3)):
        # return -np.eye(3)
        return R.from_euler("x", np.pi).as_matrix()

    K = get_skew_symmetric_matrix(np.cross(start_vec, target_vec))
    rotation = np.eye(3,3) + K + \
               np.dot(K, K) * \
               1/(1+np.dot(start_vec, target_vec) + 1E-7)
    return rotation


def signed_angle_between_vecs(target_vec, start_vec, plane_normal=None):
    # Source: https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    start_vec = np.array(start_vec)
    target_vec = np.array(target_vec)

    start_vec = start_vec/np.linalg.norm(start_vec)
    target_vec = target_vec/np.linalg.norm(target_vec)

    cross_prod = np.cross(start_vec, target_vec)

    if np.allclose(start_vec, -target_vec):
        return np.pi
    elif np.allclose(start_vec, target_vec):
        return 0

    normal_rotvec = cross_prod/np.linalg.norm(cross_prod)

    if plane_normal is None:
        arg1 = np.dot(cross_prod, normal_rotvec)
    else:
        arg1 = np.dot(cross_prod, plane_normal)
    arg2 = np.dot(start_vec, target_vec)
    # bandu_logger.debug("arg1")
    # bandu_logger.debug(arg1)
    # bandu_logger.debug("arg2")
    # bandu_logger.debug(arg2)
    return np.arctan2(arg1, arg2)


def get_rotation_between_centered_frames(frame1, frame2):
    """
    Make sure the frame vectors are normalized!!
    :param frame1: [num_points, 3]
    :param frame2: [num_points, 3]
    :return:
    """
    frame1 = np.array(frame1)
    frame2 = np.array(frame2)

    # Subtract means
    X = frame1 - np.mean(frame1, axis=0)
    Y = frame2 - np.mean(frame2, axis=0)
    # X = frame1
    # Y = frame2

    # [num_points, 3, 1]
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)


    # [num_points, 3, 1]
    H = np.sum(np.matmul(X,  np.transpose(Y, axes=[0, 2, 1])), axis=0)

    U, S, VT = np.linalg.svd(H)
    R_xy = VT.T @ U.T

    return R_xy


def umeyama(P, Q):
    P = np.array(P)
    Q = np.array(Q)
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def get_skew_symmetric_matrix(vec):
    return np.asarray([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0],
    ])


def transform_centroid_first_o3d(R_xy, start_pointcloud, start_obb, target_obb):
    """

    :param R_xy:
    :param start_pointcloud:
    :param start_obb:
    :param target_obb:
    :return: Transformed numpy pointcloud
    """
    assert start_pointcloud.shape[-1] == 3
    centered_points = (start_pointcloud - start_obb.center)

    centered_and_rotated_points = (R_xy @ centered_points.T).T

    centered_and_rotated_points = centered_and_rotated_points + target_obb.center
    return centered_and_rotated_points


def get_relative_rotation_o3d(start_frame, target_frame, ortho=True):
    """
    
    :param start_frame: [num_points, 3] (row vectors)
    :param target_frame: [num_points, 3] (row vectors)
    :return: 
    """""
    if ortho:
        U, S, VH = np.linalg.svd(target_frame.T)
        target_frame_ortho = U @ VH

        U, S, VH = np.linalg.svd(start_frame.T)
        start_frame_ortho = U @ VH
        R_xy = target_frame_ortho @ np.linalg.inv(start_frame_ortho)
    else:
        R_xy = target_frame.T @ np.linalg.inv(start_frame)

    if np.linalg.det(R_xy) < 0:
        tmp_vec = target_frame[0, :].copy()
        target_frame[0, :] = target_frame[1, :]
        target_frame[1, :] = tmp_vec

        if ortho:
            U, S, VH = np.linalg.svd(target_frame.T)
            target_frame_ortho = U @ VH

            U, S, VH = np.linalg.svd(start_frame.T)
            start_frame_ortho = U @ VH

            R_xy = target_frame_ortho @ np.linalg.inv(start_frame_ortho)
        else:
            R_xy = target_frame.T @ np.linalg.inv(start_frame.T)
    return R_xy


def sample_vector_on_sphere_uniform(theta_size=10):
    # sample on unit sphere surface
    # Convert vector into spherical coordinates

    theta = np.random.uniform(-theta_size, theta_size) # phi in the multivariable textbook
    phi = np.random.uniform(0, 180) # theta in the multivariable textbook
    r = 1

    sampled_xyz = asCartesian([r, theta, -phi])
    return sampled_xyz

def sample_vector_on_sphere_gaussian(theta_size=10):
    pass

# https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* np.pi/180 # to radian
    phi     = rthetaphi[2]* np.pi/180
    x = r * np.sin(theta ) * np.cos( phi )
    y = r * np.sin(theta ) * np.sin( phi )
    z = r * np.cos(theta )
    return np.array([x,y,z])


def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.acos(z/r)*180/ np.pi #to degrees
    phi     =  np.atan2(y,x)*180/ np.pi
    return np.array([r,theta,phi])


def apply_rot_at_origin_and_translate_back(body_id, new_rot):
    current_pos, current_rot = p.getBasePositionAndOrientation(body_id)
    new_target_quat = (R.from_quat(new_rot) * R.from_quat(current_rot)).as_quat()
    p.resetBasePositionAndOrientation(body_id, current_pos, new_target_quat)


def map_smallest_rotation(input_rotation_matrix, symmetry_matrix_arr):
    """

    :param input_rotation_matrix:
    :param symmetry_matrix_arr:
    :return: Returns Scipy Rotation object
    """
    frobenius_arr = []

    # For every element in the set of symmetries, pick the one
    for sym in symmetry_matrix_arr:
        frobenius_arr.append(eye_loss(R.from_euler("xyz", sym, degrees=True), input_rotation_matrix))
        bandu_logger.debug("sym")
        bandu_logger.debug(sym.as_euler("xyz"))

    # bandu_logger.debug("Frobenius arr")
    # bandu_logger.debug(frobenius_arr)
    min_sym_idx = np.argmin(np.array(frobenius_arr))
    bandu_logger.debug("msr min sym index")
    bandu_logger.debug(min_sym_idx)

    # bandu_logger.debug("sym_idx")
    # bandu_logger.debug(sym_idx)
    # out = (R.from_euler("xyz", symmetry_matrix_arr[sym_idx], degrees=True).inv() *R.from_matrix(input_rotation_object)).as_euler("xyz", degrees=True)
    s_inv = R.from_euler("xyz", symmetry_matrix_arr[min_sym_idx], degrees=True).inv()
    out = (s_inv
           *R.from_matrix(input_rotation_matrix))
    # bandu_logger.debug("res")
    # bandu_logger.debug(out)
    # bandu_logger.debug("\n")
    section_idx = 0
    return out, section_idx


def map_smallest_rotation_four_quadrants_two_sections(input_rotation_object):
    """

    :param input_rotation_object:
    :return: Returns Scipy Rotation object
    """

    symmetry_matrix_arr = [[0, 0, 0],
                           [0, 0, 90],
                           [0, 0, 180],
                           [0, 0, 270]]
    frobenius_arr = []

    for sym in symmetry_matrix_arr:
        loss1 = np.linalg.norm((R.from_euler("xyz", sym, degrees=True).inv() * input_rotation_object).as_matrix() - np.eye(3))
        loss2 = np.linalg.norm(R.from_euler("xyz", sym, degrees=True).as_matrix() - input_rotation_object.as_matrix())
        bandu_logger.debug("loss1")
        bandu_logger.debug(loss1)
        bandu_logger.debug("loss2")
        bandu_logger.debug(loss2)
        bandu_logger.debug("\n")
        frobenius_arr.append(loss1)

    min_sym_idx = np.argmin(np.array(frobenius_arr))

    if min_sym_idx == 0 or min_sym_idx == 1:
        s_matrix_rotation_object = R.from_euler("xyz", [0, 0, 0], degrees=True)
    else:
        s_matrix_rotation_object = R.from_euler("xyz", [0, 0, 180], degrees=True)

    if min_sym_idx == 0 or min_sym_idx == 2:
        s_section = 0
    else:
        s_section = 1

    out_rotation = s_matrix_rotation_object.inv() * input_rotation_object
    return out_rotation, s_section


def eye_loss(S, R2):
    return np.linalg.norm((S.inv() * R2).as_matrix() - np.eye(3))


# def simple_map_axis_angle(input_rotation_object, canonical_axis, angles, num_partitions=2):
# def simple_map_axis_angle(input_rotation_object, canonical_axis, angles, num_partitions=2):
#     """
#
#     :param input_rotation_object: SciPy rotation object
#     :param canonical_axis:
#     :param angles: Should be listed in order, in RADIANS
#     :param num_partitions:
#     :return:
#     """
#     canonical_axis = input_rotation_object.apply(canonical_axis)
#
#     canonical_axis = np.array(canonical_axis)/np.linalg.norm(np.array(canonical_axis))
#     assert np.isclose(np.linalg.norm(canonical_axis), 1), np.linalg.norm(canonical_axis)
#     symmetry_group_list_as_scipy_rot_objects = []
#     for angle in angles:
#         symmetry_group_list_as_scipy_rot_objects.append(R.from_rotvec(canonical_axis * angle))
#
#     # for sym in symmetry_group_list_as_scipy_rot_objects:
#     #     bandu_logger.debug("sym")
#     #     bandu_logger.debug(sym.as_euler("xyz"))
#
#     # Find index of closest symmetry group
#     # min_sym_idx = np.argmin(np.array([np.linalg.norm(s.as_matrix() - input_rotation_object.as_matrix()) for s in symmetry_group_list_as_scipy_rot_objects]))
#     min_sym_idx = np.argmin(np.array([eye_loss(s * input_rotation_object, input_rotation_object) for s in symmetry_group_list_as_scipy_rot_objects]))
#     # bandu_logger.debug("aa min sym index")
#     # bandu_logger.debug(min_sym_idx)
#     # Take index, modulo it by num partitions, to get the section index
#     sec_idx = min_sym_idx % num_partitions
#     # bandu_logger.debug("sec_idx")
#     # bandu_logger.debug(sec_idx)
#
#     # Take the symmetry rotation, undo the section rotation, then take the inverse
#     if min_sym_idx == sec_idx:
#         s_inv = (symmetry_group_list_as_scipy_rot_objects[min_sym_idx]).inv()
#     else:
#         # Get relative angle from canonical rotation to selected symmetry group rotation
#         rot_canonical = R.from_rotvec(canonical_axis*angles[sec_idx])
#         s_inv = (symmetry_group_list_as_scipy_rot_objects[min_sym_idx] * rot_canonical.inv()).inv()
#
#     # To get the new rotation we apply it to the input rotation
#     return s_inv * input_rotation_object, symmetry_group_list_as_scipy_rot_objects

# def calculate_shortest_rotation_bt_two_vectors(end_vec, start_vec):


def map_using_single_axis(input_rotation_object, single_axis, angles, num_partitions=2):
    """

    :param input_rotation_object: SciPy rotation object
    :param canonical_axis:
    :param angles: Should be listed in order, in RADIANS
    :param num_partitions:
    :return:
    """
    # single_axis = np.array([0, 0, 1])
    bandu_logger.debug("input_rotation_object")
    bandu_logger.debug(input_rotation_object.as_euler("xyz"))
    assert np.isclose(np.linalg.norm(single_axis), 1), np.linalg.norm(single_axis)
    symmetry_group_list_as_scipy_rot_objects = []
    for angle in angles:
        symmetry_group_list_as_scipy_rot_objects.append(R.from_rotvec(single_axis * angle))

    # symmetry_group_list_as_scipy_rot_objects.append(R.from_euler("xyz", [0, 0, np.pi]) * R.from_euler("xyz", [np.pi, 0, 0]))

    # Find index of closest symmetry group
    # min_sym_idx = np.argmin(np.array([np.linalg.norm(s.as_matrix() - input_rotation_object.as_matrix()) for s in symmetry_group_list_as_scipy_rot_objects]))
    bandu_logger.debug("symmetries")
    bandu_logger.debug([s.as_euler("xyz") for s in symmetry_group_list_as_scipy_rot_objects])
    losses = np.array([eye_loss(s, input_rotation_object) for s in symmetry_group_list_as_scipy_rot_objects])
    bandu_logger.debug("losses")
    bandu_logger.debug(losses)
    min_sym_idx = np.argmin(losses)
    # bandu_logger.debug("aa min sym index")
    # bandu_logger.debug(min_sym_idx)
    # Take index, modulo it by num partitions, to get the section index
    sec_idx = min_sym_idx % num_partitions
    # bandu_logger.debug("sec_idx")
    # bandu_logger.debug(sec_idx)

    # Take the symmetry rotation, undo the section rotation, then take the inverse
    if min_sym_idx == sec_idx:
        s_inv = (symmetry_group_list_as_scipy_rot_objects[min_sym_idx]).inv()
    else:
        # Get relative angle from canonical rotation to selected symmetry group rotation
        rot_canonical = R.from_rotvec(single_axis*angles[sec_idx])
        s_inv = (symmetry_group_list_as_scipy_rot_objects[min_sym_idx] * rot_canonical.inv()).inv()

    bandu_logger.debug("s_inv")
    bandu_logger.debug(s_inv.as_euler("xyz"))
    # To get the new rotation we apply it to the input rotation
    return s_inv * input_rotation_object, sec_idx, symmetry_group_list_as_scipy_rot_objects


def spherical_distance(vec1, vec2):
    # https://stackoverflow.com/questions/52210911/great-circle-distance-between-two-p-x-y-z-points-on-a-unit-sphere
    delta = np.linalg.norm(vec1 - vec2)
    phi = np.arcsin(delta/2)
    return 2* phi


def random_perturb(vec):
    rot_vec = R.random().as_rotvec()
    unit_rotvec = rot_vec/np.linalg.norm(rot_vec)
    return R.from_rotvec(unit_rotvec * .05).apply(vec)


def map_using_arbitrary_axis(input_rotation_object, canonical_axis, angles,
                             num_partitions=2, debug=False, debug2=True,
                             secondary_axis=None):
    # Axis is defined in the canonical pose of the object
    # Calculate canonical_axis in world frame
    # Calculate change of basis C (relative rotation) from canonical_axis to z-canonical_axis
    # Apply C to input rotation, and feed the result into map_using_zaxis
    # Take the output and apply C^-1 to come back to our original basis

    canonical_axis = canonical_axis / np.linalg.norm(canonical_axis)

    world_frame_axis = input_rotation_object.apply(canonical_axis)

    # if spherical_distance(world_frame_axis, canonical_axis) <= \
    #         spherical_distance(world_frame_axis, -canonical_axis):
    #     signed_canonical_axis = canonical_axis
    # else:
    #     signed_canonical_axis = -canonical_axis
    #     angles = [-ang for ang in angles]
    signed_canonical_axis = canonical_axis

    if np.all(np.isclose(world_frame_axis, -signed_canonical_axis)):
        angle = 0
    else:
        angle = signed_angle_between_vecs(signed_canonical_axis, world_frame_axis)

    # bandu_logger.debug("input rot obj")
    # bandu_logger.debug(input_rotation_object.as_euler("xyz"))
    # bandu_logger.debug("canonical axis")
    # bandu_logger.debug(signed_canonical_axis)
    # bandu_logger.debug("wfaxis")
    # bandu_logger.debug(world_frame_axis)
    # bandu_logger.debug("angle")
    #
    # bandu_logger.debug(angle)
    # bandu_logger.debug("axis")
    # bandu_logger.debug(np.cross(world_frame_axis, signed_canonical_axis))


    if np.all(np.isclose(world_frame_axis, -signed_canonical_axis)):
        assert secondary_axis is not None
        C = R.from_rotvec(np.array(secondary_axis) * np.pi)
        bandu_logger.debug("Negative world axis activated")
        # C = R.from_euler("xyz", [-np.pi, 0, 0])
    #     C = R.from_matrix(np.vstack([np.array([0, 1, 0]),
    #                                  np.array([1, 0, 0]),
    #                                 -]))
    elif np.all(np.isclose(world_frame_axis, signed_canonical_axis)):
        C = R.from_matrix(np.eye(3))
    else:
        cross_prod = np.cross(world_frame_axis, signed_canonical_axis)
        normal_rotvec = cross_prod/np.linalg.norm(cross_prod) * angle
        bandu_logger.debug("axis_angle")
        bandu_logger.debug(normal_rotvec)
        C = R.from_rotvec(normal_rotvec)

    if debug:
        ### DEBUG
        import open3d as o3d
        from bandu.utils import bandu_util, color_util
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.06)

        start_vec = np.array([1, 0, 0])

        plotted_geometries = [coordinate_frame] + [bandu_util.create_arrow(random_perturb(signed_canonical_axis), color_util.YELLOW)] + \
        [bandu_util.create_arrow(random_perturb(world_frame_axis), color_util.BLACK)] + \
        [bandu_util.create_arrow(C.apply(world_frame_axis), color_util.GOLD)]
        # [bandu_util.create_arrow(R.from_rotvec(np.cross(world_frame_axis, signed_canonical_axis) * np.pi/2).apply(world_frame_axis), color_util.TURQUOISE)]
        # [bandu_util.create_arrow(normal_rotvec, color_util.PURPLE)] + \

        o3d.visualization.draw_geometries(plotted_geometries)
        ### DEBUG


    # C = get_rotation_matrix_between_vecs(signed_canonical_axis, world_frame_axis)
    bandu_logger.debug("C euler")
    bandu_logger.debug(C.as_euler("xyz"))

    bandu_logger.debug("C applied to World frame axis ")
    bandu_logger.debug(C.apply(world_frame_axis))

    assert np.all(np.isclose(C.apply(world_frame_axis), signed_canonical_axis))

    bandu_logger.debug("Original axis")
    bandu_logger.debug(signed_canonical_axis)

    R_canonical, sec_idx, lst = map_using_single_axis(C * input_rotation_object, signed_canonical_axis, angles)
    bandu_logger.debug("\n")
    bandu_logger.debug("R_canonical")
    bandu_logger.debug(R_canonical.as_euler("zyx"))
    bandu_logger.debug("sec_idx")
    bandu_logger.debug(sec_idx)
    bandu_logger.debug("\n")

    bandu_logger.debug("C")
    bandu_logger.debug(C.as_euler("xyz"))
    if debug2:
        ### DEBUG
        import open3d as o3d
        from bandu.utils import bandu_util, color_util
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.06)

        start_vec = np.array([1, 0, 0])

        plotted_geometries = [coordinate_frame] + [bandu_util.create_arrow(random_perturb(input_rotation_object.as_rotvec()), color_util.YELLOW)] + \
                             [bandu_util.create_arrow(random_perturb((C.inv() * R_canonical).as_rotvec()), color_util.BLACK)] + \
                             [bandu_util.create_arrow(random_perturb((C * input_rotation_object).as_rotvec()), color_util.GOLD)]
        # [bandu_util.create_arrow(R.from_rotvec(np.cross(world_frame_axis, signed_canonical_axis) * np.pi/2).apply(world_frame_axis), color_util.TURQUOISE)]
        # [bandu_util.create_arrow(normal_rotvec, color_util.PURPLE)] + \

        o3d.visualization.draw_geometries(plotted_geometries)
    return C.inv() * R_canonical, C * input_rotation_object


def map_hemispheres(rotation, canonical_axes_list, canonical_vectors_list):
    if isinstance(rotation, np.ndarray) or isinstance(rotation, tuple):
        rotation = R.from_quat(rotation)
    assert len(canonical_vectors_list) > 0
    assert len(canonical_axes_list) == len(canonical_vectors_list)
    partition_indices = []
    for i in range(len(canonical_axes_list)):
        rotation, partition_idx = map_hemisphere(rotation, canonical_axes_list[i], canonical_vectors_list[i])
        partition_indices.append(partition_idx)
    assert partition_indices, partition_indices

    def partition_indices_to_integer(partition_indices):
        if len(partition_indices) == 2:
            if partition_indices[0] == 0 and partition_indices[1] == 0:
                return 0
            elif partition_indices[0] == 0 and partition_indices[1] == 1:
                return 1
            elif partition_indices[0] == 1 and partition_indices[1] == 0:
                return 2
            else:
                return 3
        elif len(partition_indices) == 1:
            return 0
        elif len(partition_indices) == 3:
            return 0
        else:
            raise NotImplementedError

    return rotation, partition_indices_to_integer(partition_indices)


def map_hemisphere(rotation, canonical_axis, canonical_vector):
    # Attach a vector to the object.
    # Track the transformed canonical axis
    # If the transformed vector is closer to the negative canonical vector than the canonical vector, we flip it around
    # Otherwise, we just keep it as is
    canonical_axis = np.array(canonical_axis)
    canonical_vector = np.array(canonical_vector)
    assert np.isclose(np.linalg.norm(canonical_axis), 1)
    assert np.isclose(np.linalg.norm(canonical_vector), 1)

    transformed_canonical_axis = rotation.apply(canonical_axis)
    transformed_vector = rotation.apply(canonical_vector)
    # if np.isclose(transformed_vector,

    if np.linalg.norm(transformed_vector - -canonical_vector) < np.linalg.norm(transformed_vector - canonical_vector):
        # Flip it over the transformed axis
        rotation = R.from_rotvec(transformed_canonical_axis * np.pi) * rotation
        bandu_logger.debug(f"Flip {canonical_axis}")
        partition_idx = 1
    else:
        bandu_logger.debug(f"No flip")
        partition_idx = 0
    return rotation, partition_idx


def get_relative_transform_world(start_quat,
                                 start_pos,
                                 target_quat,
                                 target_pos,
                                 collapse_180_around_z=False,
                                 map_function=None,
                                 map_function_kwargs=None):
    """

    :param start_homo:
    :param target_homo:
    :return: relative quat wrt world, relative trans wrt world
    """

    if map_function is not None:
        assert map_function_kwargs is not None

    # if collapse_180_around_z:
        # assert map_function is not None
        # map_fn = torch_util.str2obj("bandu.utils.transform_util." + map_function)
        # if map_function_kwargs is None:
        #     map_function_kwargs = dict()
        #
        # start_quat_obj, section_idx = map_fn(R.from_quat(start_quat), **map_function_kwargs)
        # start_quat = start_quat_obj.as_quat()
        # bandu_logger.debug("Start quat after")
        # bandu_logger.debug(start_quat)

    target_homo = homogenize_matrix(target_quat, target_pos)
    start_homo = homogenize_matrix(start_quat, start_pos)

    relative_homo_wrt_world = np.matmul(target_homo, np.linalg.inv(start_homo))
    relative_quat_world = R.from_matrix(relative_homo_wrt_world[:3, :3]).as_quat()

    partition_indices = None

    if map_function:
        relative_quat_world_rotation_object, partition_indices = map_function(relative_quat_world, **map_function_kwargs)
        relative_quat_world = relative_quat_world_rotation_object.as_quat()

    relative_trans_world = relative_homo_wrt_world[:3, 3]
    return relative_quat_world, relative_trans_world

def get_target_transform_from_relative_centroid_first(relative_trans_world_single,
                                                      relative_quat_world_single,
                                                      start_trans_single,
                                                      start_quat_single):
    target_pos_from_relative_single = relative_trans_world_single + R.from_quat(relative_quat_world_single).apply(start_trans_single)

    target_quat_from_relative_single = (R.from_quat(relative_quat_world_single)*R.from_quat(start_quat_single)).as_quat()
    return target_pos_from_relative_single, target_quat_from_relative_single


def get_target_transform_from_body(start_trans, start_quat, relative_quat_body, relative_trans_body):
    x1 = start_trans
    x2 = R.from_quat(relative_quat_body).apply(x1) + relative_trans_body
    target_pos = R.from_quat(start_quat).apply(x2) + start_trans
    target_quat = (R.from_quat(start_quat) * R.from_quat(relative_quat_body) * R.from_quat(start_quat)).as_quat()
    return target_pos, target_quat


def get_target_transform_in_world_frame(start_trans, start_quat, relative_quat_world, relative_trans_world):
    target_pos_from_relative_single = R.from_quat(relative_quat_world).apply(start_trans) + relative_trans_world
    target_quat_from_relative_single = (R.from_quat(relative_quat_world) * R.from_quat(start_quat)).as_quat()
    return target_pos_from_relative_single, target_quat_from_relative_single


def new_mapping_operator(rotation, canonical_axis, thetas):
    bandu_logger.debug("CA")
    bandu_logger.debug(canonical_axis)
    bandu_logger.debug("thetas")
    bandu_logger.debug(thetas)
    assert not isinstance(canonical_axis[0], list)

    # Construct symmetry group
    sym_rots = [R.from_rotvec(np.array(canonical_axis) * thetas[i]) for i in range(len(thetas))]

    # Find closest symmetry rotation
    s_hat_idx = np.argmin([np.linalg.norm(s.as_matrix() - rotation.as_matrix()) for s in sym_rots])
    bandu_logger.debug("s_hat_idx")
    bandu_logger.debug(s_hat_idx)
    bandu_logger.debug("\n")

    s_hat = sym_rots[s_hat_idx]

    # Apply symmetry rotation in body frame to get new rotation
    partition_idx = 0
    return rotation * s_hat, partition_idx


def new_mapping_operator_double_partition(rotation, canonical_axis, thetas):
    """

    :param rotation:
    :param canonical_axis:
    :param thetas:
    :return:
    """
    # bandu_logger.debug("CA")
    # bandu_logger.debug(canonical_axis)
    # bandu_logger.debug("thetas")
    # bandu_logger.debug(thetas)
    assert isinstance(canonical_axis[0], list), canonical_axis
    canonical_axis = canonical_axis[0]
    # double_thetas = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    double_thetas = []

    bandu_logger.debug("rotation as euler")
    bandu_logger.debug(rotation.as_euler("xyz"))

    for idx in range(len(thetas)):
        theta = thetas[idx]
        double_thetas.append(theta)
        if idx < len(thetas) - 1:
            double_thetas.append((thetas[idx + 1] - theta)/2)
        else:
            double_thetas.append((2*np.pi - theta)/2 + theta)


    bandu_logger.debug("double thetas")
    bandu_logger.debug(double_thetas)

    # Construct symmetry group
    omega_rots = [R.from_rotvec(np.array(canonical_axis) * double_thetas[i]) for i in range(len(double_thetas))]

    m_rots = [R.from_rotvec(np.array(canonical_axis) * thetas[i]) for i in range(len(thetas))]

    # Find closest symmetry rotation
    omega_idx = np.argmin([np.linalg.norm(s.as_matrix() - rotation.as_matrix()) for s in omega_rots])
    bandu_logger.debug("omega_idx")
    bandu_logger.debug(omega_idx)
    bandu_logger.debug("\n")

    # Apply symmetry rotation in body frame to get new rotation
    num_partitions = 2
    partition_idx = omega_idx % num_partitions

    bandu_logger.debug("partition idx")
    bandu_logger.debug(partition_idx)
    # if partition_idx == 0:
    # Calculate the argmin rotation with loss fn || R S - S_canonical ||
    omega_superscript = omega_rots[partition_idx]
    operator_idx = np.argmin([np.linalg.norm(rotation.as_matrix() @ s.as_matrix() - omega_superscript.as_matrix()) for s in m_rots])

    operator = m_rots[operator_idx]
    return rotation * operator, partition_idx



def new_mapping_operator_multiaxises(rotation, canonical_axises=None, theta_lists=None):
    # As long as the partitions are evenly spaced, they are good partitions
    # Which partitions map to which? Is the partition around the identity enough to cover the full space of predicted_rotations_quat?
    # As long as we apply symmetric body frame predicted_rotations_quat, we will certainly never get the incorrect image
    # Which partition will it land in?
    # If we have two thetas, we need two partitions
    # If we have two sets of two thetas each, we need 4 partitions
    assert isinstance(canonical_axises, list)
    bandu_logger.debug("\n\n")
    bandu_logger.debug("new map")

    if isinstance(rotation, np.ndarray):
        rotation = R.from_quat(rotation)
    elif isinstance(rotation, tuple):
        rotation = R.from_quat(rotation)
        # bandu_logger.debug(rotation)
        # assert NotImplementedError
    elif isinstance(rotation, R):
        pass
    else:
        raise NotImplementedError
    cartesian_product = list(itertools.product(canonical_axises, theta_lists))

    sym_rots = [R.from_rotvec(np.array(cartesian_product[i][0]) * cartesian_product[i][1]) for i in range(len(cartesian_product))]
    # Find closest symmetry rotation
    bandu_logger.debug("sym_rots")
    bandu_logger.debug(sym_rots)
    rot_mat = rotation.as_matrix()
    opt_arr = []

    for s in sym_rots:
        s_mat = s.as_matrix()
        sub = s_mat - rot_mat
        norm = np.linalg.norm(sub)
        opt_arr.append(norm)
    s_hat_idx = np.argmin(opt_arr)
    # s_hat_idx = np.argmin([np.linalg.norm(s.as_matrix() - rot_mat) for s in sym_rots])
    bandu_logger.debug("s_hat_idx")
    bandu_logger.debug(s_hat_idx)
    bandu_logger.debug("\n")

    s_hat = sym_rots[s_hat_idx]

    # Apply symmetry rotation in body frame to get new rotation
    partition_idx = 0
    return rotation * s_hat, partition_idx