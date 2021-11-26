from bandu.config import *
import os
import numpy as np

def parse_urdf_xml_for_object_name(urdf_xml_path):
    from lxml import etree

    if urdf_xml_path[0] == "/":
        tree = etree.parse(urdf_xml_path)
    else:
        try:
            tree = etree.parse(urdf_xml_path)
        except:
            tree = etree.parse(str(BANDU_ROOT) + "/" + urdf_xml_path)
    return tree.find("name").text

def get_object_names(urdf_paths_as_tuple):
    # returns object names, in same order as input tuple
    object_names = []
    for urdf_path in urdf_paths_as_tuple:
        object_name = parse_urdf_xml_for_object_name(os.path.dirname(urdf_path) + "/model.config")
        object_names.append(object_name)
    return object_names

import random
from scipy.spatial.transform import Rotation as R
def get_initial_orientation(start_rotation_type, **kwargs):
    if start_rotation_type == "full":
        initial_orientation = R.random()
    elif start_rotation_type == "planar":
        initial_orientation = R.from_quat(random_z_axis_orientation(2 * np.pi)[0])
    elif start_rotation_type == "identity":
        initial_orientation = R.from_euler("xyz", [0,0,0])
    elif start_rotation_type == "flat_or_up":
        initial_orientation = random.choice([R.from_euler("xyz", [0 ,0 ,0]), R.from_euler("xyz", [np.pi/2, 0, 0])])
    elif start_rotation_type == "close_to_target":
        initial_orientation = R.from_rotvec(misc_util.random_three_vector() * np.random.uniform(low=0.0, high=(2*np.pi/10))) \
                              * R.from_quat(kwargs['target_quat_original_order'][kwargs['obj_idx']])
    else:
        raise NotImplementedError
    return initial_orientation


def gen_2d_xy(x_min=TABLE_X_MIN, x_max=TABLE_X_MAX, y_min=TABLE_Y_MIN, y_max=TABLE_Y_MAX):
    """
    Randomize object positions such that COM is on the table.
    :return: Numpy array
    """
    i = 0
    X_sampled = np.random.uniform(x_min, x_max)
    Y_sampled = np.random.uniform(y_min, y_max)
    start_pos = np.zeros(2)
    start_pos[0] = X_sampled
    start_pos[1] = Y_sampled

    print("ln81 start_pos")
    print(start_pos)
    return start_pos


def get_position_for_drop_above_table(height_off_set=.05, avoid_center_amount=.3):
    ret_arr = np.zeros(3)

    if avoid_center_amount:
        valid_xy = gen_2d_xy()
        while np.linalg.norm(valid_xy - np.zeros(2)) < avoid_center_amount:
            valid_xy = gen_2d_xy()
        ret_arr[:2] = valid_xy
    else:
        ret_arr[:2] = gen_2d_xy()
    ret_arr[2] = TABLE_HEIGHT + height_off_set
    return ret_arr

def get_position_for_drop_above_table_avoid_obj(avoid_obj, height_off_set=.05):
    ret_arr = np.zeros(3)

    valid_xy = gen_2d_xy()
    ret_arr[2] = TABLE_HEIGHT + height_off_set

    while bool(p.getContactPoints(avoid_obj, pb_oid)):
        valid_xy = gen_2d_xy()

    ret_arr[:2] = valid_xy

    return ret_arr
