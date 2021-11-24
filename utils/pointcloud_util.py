import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.pointnet2_utils import farthest_point_sample
from utils import misc_util
import tqdm
import random
import os
import json
from deco import *


class PointcloudSampler:
    def __init__(self, num_points,
                 num_fps_samples,
                 urdf_name_to_pc_path="out/canonical_pointclouds/urdf_name_to_pointcloud_dict"):
        """
        The purpose of this object is simply to cache the farthest point sampling

        :param num_points:
        :param num_fps_samples: Number of FPS pointclouds to genereate
        :param urdf_name_to_pc_path:
        """
        self.num_points = num_points
        self.num_fps_samples = num_fps_samples
        print(f"Caching FPS pointclouds... num_points: {num_points}")
        with open(urdf_name_to_pc_path, "rb") as fp:
            self.urdf_name_to_pointcloud_dict = pickle.load(fp)
        self.urdf_name_to_fps_pointcloud = dict()
        self.urdf_name_to_fps_indices = dict()

        fill_farthestpointsampling_dict(self.urdf_name_to_fps_pointcloud,
                                        self.urdf_name_to_fps_indices,
                                        self.urdf_name_to_pointcloud_dict, self.num_fps_samples, self.num_points)

        # assert not np.array_equal(self.urdf_name_to_fps_pointcloud['Egg'][0], self.urdf_name_to_fps_pointcloud['Egg'][1])



        with open(os.path.join(os.path.dirname(urdf_name_to_pc_path), "args.json"), "r") as fp:
            self.args = json.load(fp)


    def get_canonicalized_pointclouds_ndarray(self, urdf_names):
        # returns a copy of the canonicalized pointclouds
        # print("ln20 urdf names")
        # print(urdf_names)
        # print("urdf keys")
        # print(self.urdf_name_to_fps_pointcloud.keys())
        assert not isinstance(urdf_names, str), type(urdf_names)
        # gets the pointclouds batched as an array for a given timestep, over all objects
        # all pointclouds are the SAME SIZE due to num_points_sampled
        # pointcloud_lst = [get_farthest_point_sampled_pointcloud(urdf_name_to_pointcloud_dict[name], num_points_sampled) for name in urdf_names]
        pointcloud_lst = [random.choice(self.urdf_name_to_fps_pointcloud[name]) for name in urdf_names]
        return np.stack(misc_util.pad_same_size(pointcloud_lst)).copy()


def get_farthest_point_sampled_pointcloud(pointcloud, num_points_to_sample, normals=None):
    if normals is not None:
        assert pointcloud.shape == normals.shape
    # assumes pointcloud is nB, num_points, 3

    # if this fails, we didn't collect more than 1 pointcloud...
    # assert len(pointcloud.shape) == 3, pointcloud.shape

    np.random.shuffle(pointcloud)
    fps_indices = farthest_point_sample(pointcloud, num_points_to_sample)[0].data.cpu().numpy()

    if len(pointcloud.shape) == 3:
        # if we have a batch dimension.. index into the next dim
        return pointcloud[:, fps_indices, :], normals[:, fps_indices, :]
    else:
        # return pointcloud[fps_indices, :], normals[fps_indices, :]
        return pointcloud[fps_indices, :]

import torch
# def get_fps_indices_concurrent(pointcloud, num_points_to_sample, tuple_idx, sample_idx):
#     torch.manual_seed(sample_idx)
#     random.seed(sample_idx)
#     np.random.seed(sample_idx)
#     # assumes pointcloud is nB, num_points, 3
#     np.random.shuffle(pointcloud)
#     fps_indices = farthest_point_sample(pointcloud, num_points_to_sample)[0].data.cpu().numpy()
#
#     if len(pointcloud.shape) == 3:
#         # if we have a batch dimension.. index into the next dim
#         return pointcloud[:, fps_indices, :]
#     else:
#         return pointcloud[fps_indices, :]
@concurrent
def get_fps_indices_concurrent(pointcloud, num_points_to_sample, sample_idx):
    torch.manual_seed(sample_idx)
    random.seed(sample_idx)
    np.random.seed(sample_idx)
    # assumes pointcloud is nB, num_points, 3
    np.random.shuffle(pointcloud)
    fps_indices = farthest_point_sample(pointcloud, num_points_to_sample)[0].data.cpu().numpy()

    # if len(pointcloud.shape) == 3:
    #     # if we have a batch dimension.. index into the next dim
    #         return pointcloud[:, fps_indices, :]
    # else:
    #     return pointcloud[fps_indices, :]
    return fps_indices

def get_fps_indices(pointcloud, num_points_to_sample):
    # assumes pointcloud is nB, num_points, 3
    np.random.shuffle(pointcloud)
    fps_indices = farthest_point_sample(pointcloud, num_points_to_sample)[0].data.cpu().numpy()
    return fps_indices

def fill_farthestpointsampling_dict(fps_pc_dict, fps_indices_dict, pc_dict, num_fps_samples, num_points):
    for name, tup_list in tqdm.tqdm(pc_dict.items()):
        fps_pc_ls, fps_idx_ls = get_fps_list_for_specific_name(name, pc_dict, num_fps_samples, num_points)
        fps_pc_dict[name] = fps_pc_ls
        assert fps_idx_ls
        fps_indices_dict[name] = fps_idx_ls
    return fps_pc_dict

def get_fps_list_for_specific_name(name, urdf_name_to_pointcloud_dict, num_fps_samples, num_points,
                                   noisy=False):
    """

    :param name:
    :param urdf_name_to_pointcloud_dict:
    :param num_fps_samples:
    :param num_points:
    :return: List of FPS pc, list of FPS indices
    """
    # we can't wrap this unless we reduce the two for loops into one...
    print(f"ln83 getting fps list: {name}")
    fps_pc_lst = []
    fps_indices_lst = []
    # iterate over the list of tuples associated with a name
    for name_idx in range(len(urdf_name_to_pointcloud_dict[name])):
        print(f"ln87 working on idx... {name_idx}")
        # grab a tuple, and get the 0th element of the tuple, the pc
        # pc = np.expand_dims(urdf_name_to_pointcloud_dict[name][idx][0], axis=0)
        data_tuple = urdf_name_to_pointcloud_dict[name][name_idx]

        # sample_idx noisy pointcloud
        if noisy:
            pc = 0
        else:
            pc = data_tuple[0]

        fps_samples_for_a_given_tuple = []
        fps_indices_for_a_given_tuple = []
        # iterate over the number of FPS samples
        for sample_idx in range(num_fps_samples):
            print(f"ln94 working on idx2... {sample_idx}")
            # fps_samples_for_a_given_tuple.append(get_farthest_point_sampled_pointcloud(pc, num_points,
            #                                                                            normals=data_tuple[-1] if len(data_tuple) == 4 else None))
            fps_indices = get_fps_indices(pc, num_points)

            # assert fps_indices
            if len(pc.shape) == 3:
                # if we have a batch dimension.. index into the next dim
                fps_samples_for_a_given_tuple.append(pc[:, fps_indices, :])
            else:
                fps_samples_for_a_given_tuple.append(pc[fps_indices, :])
            fps_indices_for_a_given_tuple.append(fps_indices)
        fps_pc_lst.append(fps_samples_for_a_given_tuple)
        fps_indices_lst.append(fps_indices_for_a_given_tuple)
    # assert fps_indices_lst
    return fps_pc_lst, fps_indices_lst

import itertools
@synchronized
def get_fps_list_sync_compat(name, urdf_name_to_pointcloud_dict, num_fps_samples, num_points):
    # we can't wrap this unless we reduce the two for loops into one...
    print(f"ln83 getting fps list: {name}")
    out_lst = [[None] * num_fps_samples] * len(urdf_name_to_pointcloud_dict[name])
    # iterate over the list of tuples associated with a name
    for tuple_idx, fps_sample_idx in itertools.product(range(len(urdf_name_to_pointcloud_dict[name])), range(num_fps_samples)):
        # print(f"ln87 working on idx... {idx1}")
        # grab a tuple, and get the 0th element of the tuple, the pc
        pc = urdf_name_to_pointcloud_dict[name][tuple_idx][0]

        # fps_samples_for_a_given_tuple =
        # iterate over the number of FPS samples
        # for idx2 in range(num_fps_samples):
        #     print(f"ln94 working on idx2... {idx2}")
        #     fps_samples_for_a_given_tuple.append(get_farthest_point_sampled_pointcloud(pc, num_points))
        # self.urdf_name_to_fps_pointcloud[name].append(fps_samples_for_a_given_tuple)
        # out_lst.append(fps_samples_for_a_given_tuple)
        # out_lst[tuple_idx][fps_sample_idx] = get_farthest_point_sampled_pointcloud(pc, num_points, tuple_idx, fps_sample_idx)
        out_lst[tuple_idx][fps_sample_idx] = get_fps_indices_concurrent(pc, num_points, tuple_idx, fps_sample_idx)
    return out_lst


def scale_aug_pointcloud(pc, quat, max_z_scale=2, min_z_scale=.5):
    """

    :param pc:
    :param quat:
    :param max_z_scale:
    :param min_z_scale:
    :return: Transformed PC, linear transformation matrix
    """
    # pc: num_points, 3
    assert len(pc.shape) == 2
    # assumes canonicalized pointcloud is in upright pose
    # augment the scale, ensuring that the XY scaling up goes no larger than the Z-axis height

    # rotate to canonical pose
    canonicalized_pc = R.from_quat(quat).inv().apply(pc)

    obj_width = np.max(canonicalized_pc[:, 0]) - np.min(canonicalized_pc[:, 0])
    obj_length = np.max(canonicalized_pc[:, 1]) - np.min(canonicalized_pc[:, 1])
    obj_height = np.max(canonicalized_pc[:, 2]) - np.min(canonicalized_pc[:, 2])

    z_scale = np.random.uniform(min_z_scale, max_z_scale)

    scaled_obj_height = obj_height * z_scale

    # x_height may not be
    x_scale = np.random.uniform(.5, np.max([scaled_obj_height/obj_width, .5]))

    # print("ln219 x_scale")
    # print(x_scale)

    y_scale = np.random.uniform(.5, np.max([scaled_obj_height/obj_length, .5]))
    # print("ln222 y_scale")
    # print(y_scale)


    M = np.diag(np.array([x_scale, y_scale, z_scale]))

    canonicalized_pc = (M @ canonicalized_pc.T).T

    # canonicalized_pc[:, 2] = canonicalized_pc[:, 2] * z_scale
    # canonicalized_pc[:, 0] = canonicalized_pc[:, 0] * x_scale
    # canonicalized_pc[:, 1] = canonicalized_pc[:, 1] * y_scale

    # reverse initial canonicalization
    scaled_pc = R.from_quat(quat).apply(canonicalized_pc)
    return scaled_pc, M


def shear_aug_pointcloud(pc, quat, max_shear=.5):
    canonicalzed_pc = R.from_quat(quat).inv().apply(pc.copy())

    M = np.eye(3)

    if np.random.uniform() > .5:
        # first, randomly select whether we will do XY or YX shear
        M[0, 1] = np.random.uniform(-max_shear, max_shear) #XY shear
    else:
        M[1, 0] = np.random.uniform(-max_shear, max_shear)

    canonicalzed_pc = (M @ canonicalzed_pc.T).T

    scaled_pc = R.from_quat(quat).apply(canonicalzed_pc)
    return scaled_pc, M


def augment_depth_realsense(depth, depth_adjustment=0, coefficient_scale=1):
    # augment depth according to the empirical realsense paper
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8768489&tag=1

    print("ln250 augment depths")
    print(np.max(depth))
    print(np.min(depth))
    sigma = .001063*coefficient_scale + coefficient_scale*.0007278*(depth-depth_adjustment) \
            + coefficient_scale*.003949*(depth-depth_adjustment)**2
    new_depth = np.random.normal(loc=depth, scale=sigma)

    return new_depth