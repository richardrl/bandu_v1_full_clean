import pybullet as p
from collections import namedtuple
import math
from bandu_v1_full_clean.config import TABLE_HEIGHT
from bandu_v1_full_clean.utils.vr_util import load_pr2_gripper
import numpy as np

# def load_robotiq_2f140_gripper():
#     occluded_obj_id = p.loadURDF("/home/richard/improbable/spinningup/parts/urdfs/robots/2f140_pybullet.urdf")
#     return occluded_obj_id

class PR2Gripper:
    def __init__(self, init_pos=np.array([0., 0., TABLE_HEIGHT])):
        self.pb_id = load_pr2_gripper(gripper_pos=init_pos)[0]
        # create constraint that the gripper can't move while grasping
        child_pos, child_quat = p.getBasePositionAndOrientation(self.pb_id)
        self.body_pose_constraint = p.createConstraint(self.pb_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                 child_pos, [0, 0, 0, 1], child_quat)
        self.translation_to_fingers_axis = .27

    def move_gripper(self, open_length):
        # before moving, we record the 'resetted' position before forward simulation
        cpos, cquat = p.getBasePositionAndOrientation(self.pb_id)

        self.translate(cpos)

        gripper_max_joint = 0.550569
        fraction_close = .9

        p.setJointMotorControl2(self.pb_id,
                                0,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_max_joint - fraction_close * gripper_max_joint,
                                force=10000.0) # default 1.0
        p.setJointMotorControl2(self.pb_id,
                                2,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=gripper_max_joint - fraction_close * gripper_max_joint,
                                force=10000.0) # default 1.1

    def translate(self, pos):
        cpos, cquat = p.getBasePositionAndOrientation(self.pb_id)
        p.changeConstraint(self.body_pose_constraint, pos, cquat)

from bandu_v1_full_clean.config import BANDU_ROOT

class Robotiq2F140:
    def __init__(self,
                 init_pos=np.array([0., 0., TABLE_HEIGHT + .1]),
                 urdf_path=str(BANDU_ROOT / "parts/urdfs/robots/2f140_pybullet.urdf")):
        self.pb_id = p.loadURDF(urdf_path,
                                basePosition=init_pos,
                                useFixedBase=False)
        self.mimic_children_names2coefficients = {'right_outer_knuckle_joint': -1,
                                                      'left_inner_knuckle_joint': -1,
                                                      'right_inner_knuckle_joint': -1,
                                                      'left_inner_finger_joint': 1,
                                                      'right_inner_finger_joint': 1}
                                                  # 'left_outer_knuckle_joint': -1}
        self.__parse_joint_info__()
        mimic_parent_name = 'finger_joint'
        self.__setup_mimic_joints__(mimic_parent_name, self.mimic_children_names2coefficients)
        child_pos, child_quat = p.getBasePositionAndOrientation(self.pb_id)
        self.body_pose_constraint = p.createConstraint(self.pb_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                       child_pos, [0, 0, 0, 1], child_quat)
        self.translation_to_fingers_axis = .18
        p.changeDynamics(self.pb_id,
                                self.joint2id['left_inner_finger_pad_joint'],
                                lateralFriction=2.0,
                                spinningFriction=1.0,
                                rollingFriction=1.0)
        p.changeDynamics(self.pb_id,
                                self.joint2id['right_inner_finger_pad_joint'],
                                lateralFriction=2.0,
                                spinningFriction=1.0,
                                rollingFriction=1.0)

    def __parse_joint_info__(self):
        """
        Loads joint info into self.joints_infos
        :return:
        """
        numJoints = p.getNumJoints(self.pb_id)
        jointInfo = namedtuple('jointInfo',
                               ['pb_id','name','type','damping','friction',
                                'lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints_infos = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.pb_id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.pb_id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                             jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints_infos.append(info)
        self.joint2id = {info.name: info.pb_id for info in self.joints_infos}

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.pb_id for joint in self.joints_infos if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.pb_id: self.mimic_children_names2coefficients[joint.name]
                                       for joint in self.joints_infos if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.pb_id, self.mimic_parent_id,
                                   self.pb_id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=10000, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        child_pos, child_quat = p.getBasePositionAndOrientation(self.pb_id)
        p.changeConstraint(self.body_pose_constraint, child_pos, child_quat)

        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation

        # Control the mimic gripper joint(s)
        # p.setJointMotorControl2(self.pb_id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
        #                         force=self.joints_infos[self.mimic_parent_id].maxForce,
        #                         maxVelocity=self.joints_infos[self.mimic_parent_id].maxVelocity)
        p.setJointMotorControl2(self.pb_id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=100000,
                                maxVelocity=self.joints_infos[self.mimic_parent_id].maxVelocity)

    def translate(self, pos, cquat):
        """

        :param pos: Target pos
        :param cquat: Quaternion orientation to be maintained
        :return:
        """
        p.changeConstraint(self.body_pose_constraint, pos, cquat)


import time
if __name__ == "__main__":
    p.connect(p.GUI)
    # gripper_id = load_robotiq_2f140_gripper()

    gripper_class = Robotiq2F140()

    gripper_class.move_gripper(0)

    pb_util.pb_key_loop('n')
    while 1:
        # pb_util.pb_key_loop('n')
        time.sleep(.1)
        p.stepSimulation()

# generate_samples_from_canonical_pointclouds finger joint can't move
# uvd_to_sample_on_disk finger joint moves, but it shifts the gripper pads (not desirable)
