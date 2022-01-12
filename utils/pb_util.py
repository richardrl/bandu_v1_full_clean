from bandu.config import TABLE_HEIGHT
import os
import glob
from pathlib import Path
from utils import bandu_util
import numpy as np


def get_abbs(pybullet_oid):
    num_links = p.getNumJoints(pybullet_oid)
    aabbs = []
    for link_idx in range(-1, num_links):
        aabbs.append(p.getAABB(pybullet_oid, link_idx))
    return aabbs


import time
def pb_key_loop(key):
    """
    Wait for key before breaking a while loop
    :param key:
    :return:
    """
    while 1:
        keys = p.getKeyboardEvents()
        if ord(key) in keys and keys[ord(key)] & p.KEY_WAS_RELEASED:
            break
        time.sleep(.1)


def get_object_max_height(pybullet_obj_id, ret_aabbs=False):
    aabbs = get_abbs(pybullet_obj_id)
    max_height = max([aabb[1][-1] for aabb in aabbs])
    min_height = min([aabb[0][-1] for aabb in aabbs])

    if ret_aabbs:
        assert isinstance(aabbs, list)
        return max_height, aabbs
    else:
        return max_height


def get_urdf_paths(urdf_dir):
    # Gets urdf paths and sorts according to urdf object NAMES
    # should return the same result regardless of filesystem
    urdf_paths = [f for f in glob.glob(str(Path(urdf_dir) / "**/*.urdf"), recursive=True)]
    urdf_paths += [f for f in glob.glob(str(Path(urdf_dir) / "**/*.URDF"), recursive=True)]
    assert urdf_paths, f"You are trying to load from {urdf_dir}, are you sure this is correct?"
    names = bandu_util.get_object_names(urdf_paths)
    sz = sorted(zip(urdf_paths, names), key=lambda tup: tup[1])
    sorted_urdf_paths, sorted_urdf_names = zip(*sz)
    return sorted_urdf_paths


import pybullet as p
def calculate_object_volumes(bandu_oids):
    volumes = []
    for pb_oid in bandu_oids:
        # get list of aabbs
        ls_aabbs = get_abbs(pb_oid)

        # calculate tvolume per list
        volumes.append(calculate_total_volume(ls_aabbs))
    return volumes

def calculate_total_volume(list_of_aabbs):
    # calculates the total volume based on the aabb
    total_volume = 0
    for aabb in list_of_aabbs:
        min_vec = aabb[0]
        max_vec = aabb[1]

        single_aabb_volume = 1
        for axis in range(3):
            len_along_axis = np.linalg.norm(max_vec[axis] - min_vec[axis])
            single_aabb_volume *= len_along_axis
        total_volume += single_aabb_volume
    return total_volume


def get_stl_paths(sorted_urdf_paths, stl_dir):
    assert stl_dir is not None
    # get name of each
    output_stls = []
    names = bandu_util.get_object_names(sorted_urdf_paths)
    for name in names:
        if os.path.isfile(Path(stl_dir) / f"{name}.stl"):
            output_stls.append(str(Path(stl_dir) / f"{name}.stl"))
        elif os.path.isfile(Path(stl_dir) / f"{name}.STL"):
            output_stls.append(str(Path(stl_dir) / f"{name}.STL"))
        else:
            print("ln422 stl path")
            print(Path(stl_dir) / f"{name}.STL")
            raise NotImplementedError
    return output_stls


def get_object_height(pybullet_obj_id, ret_aabbs=False):
    aabbs = get_abbs(pybullet_obj_id)
    max_height = max([aabb[1][-1] for aabb in aabbs])
    min_height = min([aabb[0][-1] for aabb in aabbs])

    if ret_aabbs:
        assert isinstance(aabbs, list)
        return max_height-min_height, aabbs
    else:
        return max_height-min_height


def get_multiobj_aa_bounds(lst_of_pb_obj_ids, axis):
    total_max = float("-inf")
    total_min = float("+inf")
    for pybullet_obj_id in lst_of_pb_obj_ids:
        aabbs = get_abbs(pybullet_obj_id)
        max_ = max([aabb[1][axis] for aabb in aabbs])
        min_ = min([aabb[0][axis] for aabb in aabbs])

        if max_ > total_max:
            total_max = max_
        if min_ < total_min:
            total_min = min_
    return total_min, total_max

def get_object_aa_length(pybullet_obj_id, axis, ret_aabbs=False):
    aabbs = get_abbs(pybullet_obj_id)
    max_height = max([aabb[1][axis] for aabb in aabbs])
    min_height = min([aabb[0][axis] for aabb in aabbs])

    if ret_aabbs:
        assert isinstance(aabbs, list)
        return max_height-min_height, aabbs
    else:
        return max_height-min_height


def get_tower_height(pb_object_ids, keep_table_height=False, scale=True,
                     ret_max_height_id=False):
    """

    :param pb_object_ids:
    :param keep_table_height:
    :param scale:
    :param ret_max_height_id: Return object ID at the max height
    :return:
    """
    # gets top height of all objects in the scene
    # in cm
    if not pb_object_ids:
        if ret_max_height_id:
            return TABLE_HEIGHT, 1
        else:
            return TABLE_HEIGHT
    overall_top = -float("inf")

    tallest_obj_id = None
    for poid in pb_object_ids:
        object_height = get_object_max_height(poid)

        if object_height > overall_top:
            tallest_obj_id = poid
            # overall_top = max(overall_top, object_height)
            overall_top = object_height

    candidate_height = overall_top
    if keep_table_height:
        if scale:
            candidate_height = candidate_height * 100
    else:
        candidate_height = candidate_height - TABLE_HEIGHT

    if ret_max_height_id:
        return candidate_height, tallest_obj_id
    else:
        return candidate_height