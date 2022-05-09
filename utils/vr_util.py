import glob

import pybullet as p
import os

from bandu_v1_full_clean.config import BANDU_ROOT
from bandu_v1_full_clean.config import bandu_logger
vr_dir_path = os.path.dirname(os.path.realpath(__file__))


def load_stick_gripper():
    """
    Loads a gripper that is just a yellow. Return the body ID
    :return:
    """
    pr2_gripper_path = str(BANDU_ROOT / "parts/non_bandu/stick_gripper_urdf/stick_gripper.urdf")
    gripper_id = p.loadURDF(str(pr2_gripper_path), 0.500000, 0.300006, 0.700000, -0.000000, -0.000000, -0.000031,
               1.000000)
    return gripper_id


def load_pr2_gripper(gripper_pos=[0.500000, 0.300006, 0.700000]):
    pr2_gripper_path = str(BANDU_ROOT / "parts/non_bandu/pr2_gripper_urdf/pr2_gripper_mine.urdf")
    objects = [
        p.loadURDF(str(pr2_gripper_path), *gripper_pos, -0.000000, -0.000000, -0.000031,
                   1.000000)
    ]
    pr2_gripper = objects[0]
    bandu_logger.debug("pr2_gripper=")
    bandu_logger.debug(pr2_gripper)

    jointPositions = [0.550569, 0.000000, 0.549657, 0.000000]
    for jointIndex in range(p.getNumJoints(pr2_gripper)):
        p.resetJointState(pr2_gripper, jointIndex, jointPositions[jointIndex])
        p.setJointMotorControl2(pr2_gripper, jointIndex, p.POSITION_CONTROL, targetPosition=0, force=0)

    # import pdb
    # pdb.set_trace()
    # pr2_finger1_cid = p.createConstraint(pr2_gripper, -1, pr2_gripper, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0],
    #                              [0.500000, 0.300006, 0.700000])
    # bandu_logger.debug("pr2_cid")
    # bandu_logger.debug(pr2_finger1_cid)
    #

    # constraint to keep fingers symmetric
    pr2_finger2_cid2 = p.createConstraint(pr2_gripper,
                                  0,
                                  pr2_gripper,
                                  2,
                                  jointType=p.JOINT_GEAR,
                                  jointAxis=[0, 1, 0],
                                  parentFramePosition=[0, 0, 0],
                                  childFramePosition=[0, 0, 0])
    p.changeConstraint(pr2_finger2_cid2, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=50)
    pr2_finger1_cid = 0
    pr2_finger2_cid2 = 0
    return pr2_gripper, pr2_finger1_cid, pr2_finger2_cid2


def load_pr2_gripper_stick():
    pr2_gripper_path = str(BANDU_ROOT / "parts/non_bandu/pr2_gripper_urdf/pr2_gripper_mine.urdf")
    objects = [
        p.loadURDF(str(pr2_gripper_path), 0.500000, 0.300006, 0.700000, -0.000000, -0.000000, -0.000031,
                   1.000000)
    ]
    pr2_gripper = objects[0]
    bandu_logger.debug("pr2_gripper=")
    bandu_logger.debug(pr2_gripper)

    jointPositions = [0.550569, 0.000000, 0.549657, 0.000000]
    for jointIndex in range(p.getNumJoints(pr2_gripper)):
        p.resetJointState(pr2_gripper, jointIndex, jointPositions[jointIndex])
        p.setJointMotorControl2(pr2_gripper, jointIndex, p.POSITION_CONTROL, targetPosition=0, force=0)

    pr2_cid = p.createConstraint(pr2_gripper, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0],
                                 [0.500000, 0.300006, 0.700000])
    bandu_logger.debug("pr2_cid")
    bandu_logger.debug(pr2_cid)

    pr2_cid2 = p.createConstraint(pr2_gripper,
                                  0,
                                  pr2_gripper,
                                  2,
                                  jointType=p.JOINT_GEAR,
                                  jointAxis=[0, 1, 0],
                                  parentFramePosition=[0, 0, 0],
                                  childFramePosition=[0, 0, 0])
    p.changeConstraint(pr2_cid2, gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=3)
    return pr2_gripper, pr2_cid, pr2_cid2


def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    bandu_logger.debug("Substring not found. Did you convert STL to URDF?")
    raise Exception


def convert_windowspath_to_posixpath(windows_path_string):
    """

    :param windows_path_string:
    :return: String of absolute posix path
    """
    if windows_path_string[0:2].lower() == "c:":
        out_str = windows_path_string[2:]
        out_str = out_str.replace("\\", "/")

        start_chop_index= out_str.find("bandu/")
        out_str = out_str[start_chop_index+6:] # Strip everything before .. to make it a relative path

        out_str = BANDU_ROOT / out_str
        bandu_logger.debug("Converted windows path " +str(out_str))
        return str(out_str)
    else:
        bandu_logger.debug("Not converting this string because it's not a windows path "+ str(windows_path_string))
        return windows_path_string

from pathlib import Path
def walk_log_dir(log_dir):
    """
    Searches for all pr2_demo_vr.bin files and returns the directory containing them
    :param log_dir:
    :return: Returns absolute path to all subdirs
    """
    # return [_[0] for _ in os.walk(str(BANDU_ROOT / "vr/logs" / log_dir))][1:]
    bin_files = [f for f in glob.glob(str(Path(log_dir) / "**/pr2_demo_vr.bin"), recursive=True)]
    return [os.path.dirname(filepath) for filepath in bin_files]

def detect_vr_controller_and_load_plugin(pr2_cid, pr2_cid2, pr2_gripper):
    """
    Detects which gripper you are using.
    Starts up system. 
    :param pr2_cid:
    :param pr2_cid2:
    :param pr2_gripper:
    :return:
    """
    global TRIGGER_BUTTON
    controllerId = -1

    bandu_logger.debug("waiting for VR controller trigger")
    while (controllerId < 0):
        events = p.getVREvents()
        for e in (events):
            if (e[BUTTONS][33] == p.VR_BUTTON_IS_DOWN):
                controllerId = e[CONTROLLER_ID]
                TRIGGER_BUTTON = 33
            if (e[BUTTONS][32] == p.VR_BUTTON_IS_DOWN):
                controllerId = e[CONTROLLER_ID]
                TRIGGER_BUTTON = 32

    bandu_logger.debug("Using controllerId=" + str(controllerId))

    plugin = p.loadPlugin("vrSyncPlugin")
    bandu_logger.debug("PluginId=" + str(plugin))

    p.executePluginCommand(plugin, "bla", [controllerId, pr2_cid, pr2_cid2, pr2_gripper], [50, 3])

# CONTROLLER_ID = 0
POSITION = 1
ORIENTATION = 2
# ANALOG = 3
# BUTTONS = 6

def detect_vr_controller_and_sync_pb_gripper(pr2_cid, pr2_cid2, pr2_gripper):
    """
    Detects which gripper you are using.
    Starts up system.
    :param pr2_cid:
    :param pr2_cid2:
    :param pr2_gripper:
    :return:
    """
    global TRIGGER_BUTTON
    controllerId = -1

    bandu_logger.debug("waiting for VR controller trigger")
    while (controllerId < 0):
        events = p.getVREvents()
        for e in (events):
            if (e[BUTTONS][33] == p.VR_BUTTON_IS_DOWN):
                controllerId = e[CONTROLLER_ID]
                TRIGGER_BUTTON = 33
            if (e[BUTTONS][32] == p.VR_BUTTON_IS_DOWN):
                controllerId = e[CONTROLLER_ID]
                TRIGGER_BUTTON = 32

    bandu_logger.debug("Using controllerId=" + str(controllerId))

    # plugin = p.loadPlugin("vrSyncPlugin")
    # bandu_logger.debug("PluginId=" + str(plugin))
    #
    # p.executePluginCommand(plugin, "bla", [controllerId, pr2_cid, pr2_cid2, pr2_gripper], [50, 3])

BUTTONS = 6
CONTROLLER_ID = 0


def get_urdfs_paths_from_disk():
    urdf_dir_ = BANDU_ROOT / "parts/urdfs/**/*.urdf"
    urdfs_paths_as_strings = [f for f in glob.glob(str(urdf_dir_),
                                                   recursive=True)]
    bandu_logger.debug(f"Loaded {len(urdfs_paths_as_strings)} urdf paths")
    return urdfs_paths_as_strings


def get_objname_from_urdf_path(urdf_path_str):
    return urdf_path_str.lower().split(".urdf")[0].split("/")[-1]


def get_objnames_from_urdf_path_(list_urdf_paths):
    return tuple([get_objname_from_urdf_path(path) for path in list_urdf_paths])


from itertools import combinations, permutations
import random


def get_urdf_combinations(num_objects, shuffle=False):
    # Returns list of tuples, where each tuple is size num_objects
    # The list consists of all (num_urdfs) C (num_objects) combinations

    ret_list = list(combinations(get_urdfs_paths_from_disk(), num_objects))
    if shuffle:
        return random.sample(ret_list, len(ret_list))
    else:
        return ret_list


def get_urdf_permutations(num_objects, shuffle=False):
    # Returns list of tuples, where each tuple is size num_objects
    # The list consists of all (num_urdfs) C (num_objects) combinations

    ret_list = list(permutations(get_urdfs_paths_from_disk(), num_objects))
    if shuffle:
        return random.sample(ret_list, len(ret_list))
    else:
        return ret_list