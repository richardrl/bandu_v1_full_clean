import gym.utils.seeding
from scipy.spatial.transform import Rotation as R
# from bandu.utils import *
from bandu import config
from pathlib import Path
import random

import pybullet as p

from utils.misc_util import *
from utils.pb_util import get_object_height, get_tower_height, get_urdf_paths, get_stl_paths
from itertools import combinations_with_replacement

# from bandu.utils import camera_util
import os
import json

from utils.color_util import get_colors

from utils import bandu_util, camera_util, mesh_util, pb_util

# from imports.airobot.src.airobot.arm.single_arm_pybullet import SingleArmPybullet
from yacs.config import CfgNode as CN
from yacs.config import CfgNode


def convert_ints_to_boolean_mask(list_of_ints, max_classes):
    for int_ in list_of_ints:
        assert int_ >= 0, int_
    # turns a list of natural numbers into a boolean mask as vector
    return np.sum(np.eye(max_classes)[list_of_ints], axis=0).astype(int)


class BanduNoopEnv(gym.Env):
    def __init__(self,
                 *args,
                 reward_type=None,
                 num_objects=None,
                 urdfs=None,
                 stls=None,
                 urdf_dir=None,
                 stl_dir=None,
                 start_rotation_type=None,
                 forward_sim_steps=500,
                 cam_pkl=Path(config.BANDU_ROOT) / "pybullet/cams/front_cam0.pkl",
                 debug=False,
                 randomize_mesh_vertices=True,
                 p_connect_type=p.DIRECT,
                 realtime=False,
                 scale_factor=1.0,
                 num_sub_steps=1,
                 urdf_holdout_frac=0,
                 urdf_holdout_sampling_strategy=None,
                 phase="train",
                 pointcloud_state=False,
                 placement_offset=-.003,
                 height_offset=.4,
                 xy_noise_level=0.0,
                 pb_client = None,
                 use_arm=False,
                 gen_table_pointcloud=False
                 ):
        """

        :param args:
        :param reward_type:
        :param num_objects:
        :param urdfs:
        :param stls:
        :param urdf_dir:
        :param stl_dir:
        :param start_rotation_type:
        :param forward_sim_steps:
        :param cam_pkl:
        :param debug:
        :param randomize_mesh_vertices:
        :param p_connect_type:
        :param realtime:
        :param scale_factor:
        :param num_sub_steps:
        :param placement_offset: the amount along the z axis to offset
        :param xy_noise_level: the amount in XY to randomize, e.g. actuator noise
        :param urdf_holdout: Should you randomly sample_idx, or holdout some URDF tuples for testing
        :param pointcloud_state: Whether or not to render object pointclouds and pass it in the state.
            If we do, we must be trying to do control from pointclouds. So, assert that there is a quaternion in the
            actionspace.

        """
        # super().__init__(*args, **kwargs)
        # kwargs = train_dataset_params
        # self.dataset = BanduHandspecifiedMultiobject(*args, **dataset_kwargs)
        assert num_objects is not None
        plane_id, table_id = bandu_util.load_scene(p_connect_type=p_connect_type, realtime=realtime)

        self.plane_id = plane_id
        self.table_id = table_id

        p.setPhysicsEngineParameter(numSubSteps=num_sub_steps)

        self.p_connect_type = p_connect_type
        self.cam_pkl = cam_pkl
        self.num_objects = num_objects
        self.reward_type = reward_type
        self.forward_sim_steps = forward_sim_steps
        assert reward_type in ["height", "binary"]
        self.first_reset_completed = False
        self.debug = debug
        self.start_rotation_type = start_rotation_type

        print("ln82 urdf_dir")
        print(urdf_dir)

        self.urdf_paths = get_urdf_paths(urdf_dir)

        self.urdf_colors = get_colors(len(self.urdf_paths))
        self.stl_paths = get_stl_paths(self.urdf_paths, stl_dir)
        self.randomize_mesh_vertices = randomize_mesh_vertices
        self.scale_factor = scale_factor
        self.urdf_holdout_frac = urdf_holdout_frac
        self.pointcloud_state = pointcloud_state
        self.placement_offset = placement_offset
        if pointcloud_state:
            self.cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                                     camera_forward_z_offset=.2)
            # self.cameras = camera_util.setup_cameras()

        self.height_offset = height_offset
        self.xy_noise_level = xy_noise_level
        if urdf_holdout_frac:
            assert phase is not None
            self.phase = phase
        assert urdf_holdout_sampling_strategy in ['sequential', "one_cap", None, 'block_base']
        self.current_urdf_tuple_idx = 0
        self.urdf_tuple_indices = list(combinations_with_replacement(range(len(self.urdf_paths)), num_objects))
        self.urdf_holdout_sampling_strategy = urdf_holdout_sampling_strategy

        self.gen_table_pointcloud = gen_table_pointcloud

        if use_arm:
            cnode = CN(
                {'HAS_ARM': True, 'HAS_BASE': False, 'HAS_CAMERA': True, 'HAS_EETOOL': True, 'ARM':
                    CfgNode({'CLASS': 'UR5e',
                             'MOVEGROUP_NAME': 'manipulator',
                             'ROSTOPIC_JOINT_STATES': '/joint_states',
                             'MAX_TORQUES': [150, 150, 150, 28, 28, 28],
                             'JOINT_NAMES': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                             'ROBOT_BASE_FRAME': 'base', 'ROBOT_EE_FRAME': 'ee_tip', 'ROBOT_EE_FRAME_JOINT': 'ee_tip_joint', 'JOINT_SPEED_TOPIC': '/joint_speed', 'URSCRIPT_TOPIC': '/ur_driver/URScript', 'IK_POSITION_TOLERANCE': 0.01, 'IK_ORIENTATION_TOLERANCE': 0.05, 'HOME_POSITION': [0, -1.66, -1.92, -1.12, 1.57, 0], 'MAX_JOINT_ERROR': 0.01, 'MAX_JOINT_VEL_ERROR': 0.05, 'MAX_EE_POS_ERROR': 0.01, 'MAX_EE_ORI_ERROR': 0.02, 'TIMEOUT_LIMIT': 10, 'PYBULLET_RESET_POS': [0, 0, 1], 'PYBULLET_RESET_ORI': [0, 0, 0], 'PYBULLET_IK_DAMPING': 0.0005}), 'CAM': CfgNode({'CLASS': 'RGBDCamera', 'SIM': CfgNode({'ZNEAR': 0.01, 'ZFAR': 10, 'WIDTH': 640, 'HEIGHT': 480, 'FOV': 60}), 'REAL': CfgNode({'ROSTOPIC_CAMERA_INFO': 'camera/color/camera_info', 'ROSTOPIC_CAMERA_RGB': 'camera/color/image_rect_color', 'ROSTOPIC_CAMERA_DEPTH': 'camera/aligned_depth_to_color/image_raw', 'DEPTH_MIN': 0.2, 'DEPTH_MAX': 2, 'DEPTH_SCALE': 0.001})}), 'BASE': CfgNode({'CLASS': ''}), 'EETOOL': CfgNode({'OPEN_ANGLE': 0.0, 'CLOSE_ANGLE': 0.7, 'SOCKET_HOST': '127.0.0.1', 'SOCKET_PORT': 63352, 'SOCKET_NAME': 'gripper_socket', 'DEFAULT_SPEED': 255, 'DEFAULT_FORCE': 50, 'COMMAND_TOPIC': '/ur_driver/URScript', 'GAZEBO_COMMAND_TOPIC': '/gripper/gripper_cmd/goal', 'JOINT_STATE_TOPIC': '/joint_states', 'IP_PREFIX': '192.168', 'UPDATE_TIMEOUT': 5.0, 'POSITION_RANGE': 255, 'POSITION_SCALING': 364.28571428571433, 'JOINT_NAMES': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint'], 'MIMIC_COEFF': [1, -1, 1, -1, -1, 1], 'MAX_TORQUE': 25.0, 'CLASS': 'Robotiq2F140Pybullet'}), 'ROBOT_DESCRIPTION': '/robot_description', 'PYBULLET_URDF': '/home/richard/improbable/airobot/src/airobot/urdfs/ur5e_2f140_pybullet.urdf'}
            )

            self.arm = SingleArmPybullet(cnode, -1)
            self.arm.reset()
            self.arm._build_jnt_id()
            self.arm.go_home(ignore_physics=True)
        else:
            self.arm = None

    def seed(self, seed=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return

    def get_current_quat_and_pos(self, npify=False):
        pos_arr = []
        ori_arr = []
        for obj_id in self.pybullet_object_ids:
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            pos_arr.append(pos)
            ori_arr.append(ori)
        if npify:
            return np.array(pos_arr), np.array(ori_arr)
        else:
            return pos_arr, ori_arr

    def load_meshes_from_tuple_id(self, tuple_id):
        stl_paths_for_sample = self.stl_combos[tuple_id]
        return bandu_util.load_centered_meshes(stl_paths_for_sample)

    def update_meshes(self, canonical_mesh_vertices, quats):
        # assert canonical_mesh_vertices.dtype != np.dtype('O')
        assert isinstance(canonical_mesh_vertices, list)
        out = []
        # print(type(canonical_meshes_vertices))
        # assert isinstance(canonical_meshes_vertices, np.ndarray)
        for obj_id in range(len(self.pybullet_object_ids)):
            new_mesh_vertices = R.from_quat(quats[obj_id]).apply(canonical_mesh_vertices[obj_id])
            out.append(new_mesh_vertices)
        return out

    def sample_objects_with_replacement(self, num_objects):
        # returns set of urdf paths
        assert len(self.urdf_paths) == len(self.stl_paths), (len(self.urdf_paths), len(self.stl_paths))
        # zipped = random.choices(list(zip(self.urdf_paths, self.stl_paths)), k=num_objects)
        # return zip(*zipped)
        idxs = random.choices(range(len(self.urdf_paths)), k=num_objects)
        return [self.urdf_paths[idx] for idx in idxs], [self.stl_paths[idx] for idx in idxs], idxs

    def sample_objects_with_replacement_holdout(self):
        if self.phase == "train":
            # get a random tuple of indices
            tuple_of_idxs = random.choice(self.urdf_tuple_indices[:int(self.urdf_holdout_frac * len(self.urdf_tuple_indices))])
            for idx in tuple_of_idxs:
                assert isinstance(idx, int)
            assert len(tuple_of_idxs) == self.num_objects
        else:
            # if not train, then test
            tuple_of_idxs = random.choice(self.urdf_tuple_indices[int(self.urdf_holdout_frac * len(self.urdf_tuple_indices)):])
        return [self.urdf_paths[idx] for idx in tuple_of_idxs], [self.stl_paths[idx] for idx in tuple_of_idxs], tuple_of_idxs

    def sample_objects_sequentially_holdout(self):
        tuple_of_idxs = self.urdf_tuple_indices[min(self.current_urdf_tuple_idx, len(self.urdf_tuple_indices) - 1)]

        if self.current_urdf_tuple_idx == len(self.urdf_tuple_indices) -1:
            self.current_urdf_tuple_idx = 0
        else:
            self.current_urdf_tuple_idx += 1

        import pdb
        pdb.set_trace()
        return [self.urdf_paths[idx] for idx in tuple_of_idxs], [self.stl_paths[idx] for idx in tuple_of_idxs], tuple_of_idxs

    def sample_one_cap(self):
        # sample a bunch of supports
        # sample one cap

        # make a list of supports
        # make a list of caps

        cap_ids = []
        support_ids = []
        for idx, pth in enumerate(self.urdf_paths):
            dir_ = os.path.dirname(pth)
            config_path = Path(dir_) / "extra_config"
            with open(str(config_path), "r") as fp:
                jd = json.load(fp)
                if jd['block_type'] == "cap":
                    cap_ids.append(idx)
                elif jd['block_type'] == "support":
                    support_ids.append(idx)

        support_idxs = random.choices(support_ids, k=self.num_objects - 1)

        cap_id = random.choice(cap_ids)

        out_ids = support_idxs + [cap_id]

        return [self.urdf_paths[idx] for idx in out_ids], [self.stl_paths[idx] for idx in out_ids], out_ids

    def sample_objects_block_base(self):
        # make the first object always a foundation block
        tuple_of_idxs = self.urdf_tuple_indices[min(self.current_urdf_tuple_idx, len(self.urdf_tuple_indices) - 1)]


        if self.current_urdf_tuple_idx == len(self.urdf_tuple_indices) -1:
            self.current_urdf_tuple_idx = 0
        else:
            self.current_urdf_tuple_idx += 1

        ls = list(tuple_of_idxs)

        foundation_idx = next(i for i,v in enumerate(self.urdf_paths) if "foundation" in v)
        tuple_of_idxs = [foundation_idx]

        tuple_of_idxs += ls

        urdfs_ret = [self.urdf_paths[idx] for idx in tuple_of_idxs]
        stls_ret = [self.stl_paths[idx] for idx in tuple_of_idxs]

        # urdfs_ret.append("parts/urdfs/main/engmikedset/foundation/foundation.urdf")
        # stls_ret.append("parts/stls/main/engmikedset/foundation.stl")

        return urdfs_ret, stls_ret, tuple_of_idxs

    def get_urdfs_stls(self, tuple_of_idxs):
        return [self.urdf_paths[idx] for idx in tuple_of_idxs], [self.stl_paths[idx] for idx in tuple_of_idxs]

    def reset(self, sampled_path_idxs=None):
        if self.arm is not None:
            self.arm.go_home(ignore_physics=True)

        if self.first_reset_completed:
            bandu_util.remove_bandu_objects(self.num_objects)
        else:
            self.first_reset_completed = True

        if sampled_path_idxs is not None:
            urdf_paths_for_sample, stl_paths_for_sample = self.get_urdfs_stls(sampled_path_idxs)
        else:
            if self.urdf_holdout_sampling_strategy == "sequential":
                urdf_paths_for_sample, stl_paths_for_sample, sampled_path_idxs = self.sample_objects_sequentially_holdout()
            elif self.urdf_holdout_sampling_strategy == "block_base":
                urdf_paths_for_sample, stl_paths_for_sample, sampled_path_idxs = self.sample_objects_block_base()
            elif self.urdf_holdout_sampling_strategy == "one_cap":
                urdf_paths_for_sample, stl_paths_for_sample, sampled_path_idxs = self.sample_one_cap()
            elif self.urdf_holdout_frac:
                # with replacement
                urdf_paths_for_sample, stl_paths_for_sample, sampled_path_idxs = self.sample_objects_with_replacement_holdout()
            else:
                urdf_paths_for_sample, stl_paths_for_sample, sampled_path_idxs = self.sample_objects_with_replacement(self.num_objects)

        assert sampled_path_idxs

        self.object_names = bandu_util.get_object_names(urdf_paths_for_sample)
        print("ln295 object_names")
        print(self.object_names)
        # pybullet_object_ids represent the movable bandu objects

        # colors = [[float(it) for it in self.urdf_colors[_].split(" ")] for _ in sampled_path_idxs]
        # print("ln271 sampled_path_idxs (colors)")
        # print(sampled_path_idxs)

        # colors = []
        # for _ in sampled_path_idxs:
        #     for it in self.urdf_colors[_ % len(self.urdf_colors)]:
        #         colors.append(float(it))
        colors = [[1., 1., 1.] for _ in sampled_path_idxs]
        # colors = [[float(it) for it in self.urdf_colors[_ % len(self.urdf_colors)]] for _ in sampled_path_idxs]

        colors_alpha = [color + [1.0] for color in colors]
        self.pybullet_object_ids, self.extra_config = bandu_util.load_bandu_objects(urdf_paths=urdf_paths_for_sample,
                                                                                    scale_factor=self.scale_factor,
                                                                                    read_extra_config=True,
                                                                                    colors=colors_alpha)

        self.canonical_mesh_objects = bandu_util.load_centered_meshes(stl_paths_for_sample)

        # print("ln151 stl_paths")
        # print(stl_paths_for_sample)
        canonical_meshes_vertices = bandu_util.load_meshes_vertices(stl_paths_for_sample, self.randomize_mesh_vertices)

        # create placeholder cylinder
        # check collision -> if collide, restart
        # remove cylinder at the end
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                            rgbaColor=[1, 0, 0, 0],
                                            radius=0.2,
                                            length=2.)

        collisionId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                             radius=0.2,
                                             height=2.)

        avoid_obj_id = p.createMultiBody(baseMass=0,
                               baseCollisionShapeIndex=collisionId,
                               baseVisualShapeIndex=visualShapeId,
                               basePosition=[0., 0., 0.],
                               baseOrientation=[0., 0., 0., 1.]
                               )

        for idx, pybullet_oid in enumerate(self.pybullet_object_ids):
            def setup_obj(pb_id, avoid_obj_id):
                initial_orientation = bandu_util.get_initial_orientation(self.start_rotation_type)

                if self.start_rotation_type == "identity":
                    object_height, list_of_aabbs = get_object_height(pb_id, ret_aabbs=True)

                    start_pos = np.array([0., 0., config.TABLE_HEIGHT + object_height/2])
                else:
                    start_pos = bandu_util.get_position_for_drop_above_table(height_off_set=self.height_offset,
                                                                            avoid_center_amount=.3)

                p.resetBasePositionAndOrientation(pb_id, start_pos, initial_orientation.as_quat())

                iters = 0
                while len(p.getContactPoints(avoid_obj_id, pb_id)) > 1 and iters < 10000:
                    res = p.getContactPoints(avoid_obj_id, pb_id)
                    # p.addUserDebugText("CP", res[0][5])
                    # import pdb
                    # pdb.set_trace()
                    print("ln239 contacts")
                    print(p.getContactPoints(avoid_obj_id, pb_id))
                    initial_orientation = bandu_util.get_initial_orientation(self.start_rotation_type)
                    if self.start_rotation_type == "identity":
                        object_height, list_of_aabbs = get_object_height(pb_id, ret_aabbs=True)

                        start_pos = np.array([0., 0., config.TABLE_HEIGHT + object_height/2])
                    else:
                        start_pos = bandu_util.get_position_for_drop_above_table(height_off_set=self.height_offset,
                                                                                avoid_center_amount=.3)

                    p.resetBasePositionAndOrientation(pb_id, start_pos, initial_orientation.as_quat())
                    iters += 1

                if self.forward_sim_steps:
                    for _ in range(self.forward_sim_steps):
                        # print("ln166 forward simulating...")
                        p.stepSimulation()

            setup_obj(pybullet_oid, avoid_obj_id)

            def is_colliding_center_region(pb_oid, center_obj_id):
                truthy = bool(p.getContactPoints(center_obj_id, pb_oid))

                return truthy

            # while np.linalg.norm(np.array(pos) - np.zeros(3)) < .3:

            def is_under_table(pb_oid):
                pos, quat = p.getBasePositionAndOrientation(pb_oid)
                if pos[-1] < config.TABLE_HEIGHT - .1:
                    return True
                else:
                    return False


            while is_colliding_center_region(pybullet_oid, avoid_obj_id) and is_under_table(pybullet_oid):
                setup_obj(pybullet_oid, avoid_obj_id)

        p.removeBody(avoid_obj_id)
        if self.forward_sim_steps:
            for _ in range(self.forward_sim_steps):
                p.stepSimulation()


        self.current_centered_mesh_vertices = self.update_meshes(canonical_meshes_vertices,
                                                            self.get_current_quat_and_pos(npify=True)[1])
        # print("ln180 ccmv")
        # print(self.current_centered_mesh_vertices.shape)
        mtc_arr, mfn_arr, labels_arr, start_mtc_arr, start_mfn_arr, positive_index_arr = \
            mesh_util.update_tricenters_and_normals(self.object_names,
                                                    self.canonical_mesh_objects,
                                                    self.get_current_quat_and_pos(npify=True)[1],
                                                    npify=False)

        self.current_phace_normals = start_mfn_arr
        self.current_tri_centers = start_mtc_arr

        self.mesh_phace_normals = mfn_arr
        self.mesh_tri_centers = mtc_arr
        self.num_objects = len(self.pybullet_object_ids)

        self.timestep = np.array([0])
        self.canonical_mesh_normal_labels = labels_arr

        self.visited = set()

        self.capped = False

        self.stl_paths_for_sample = stl_paths_for_sample
        self.future_moved_from_target = False
        # self.tuple_id = np.array(tuple_id)
        self.current_pos, self.current_quats = self.get_current_quat_and_pos(npify=True)
        self.sampled_path_idxs = sampled_path_idxs

        self.object_id_move_order = []

        o = dict(current_tri_centers=self.current_tri_centers,
                 current_phace_normals=self.current_phace_normals,
                 pybullet_object_ids=np.array(self.pybullet_object_ids),  # This is almost always the same, because we delete the objects after every reset
                 timestep=self.timestep,
                 visited=convert_ints_to_boolean_mask(list(self.visited), self.num_objects),
                 extra_config=self.extra_config,
                 capped=np.asarray(self.capped),
                 current_centered_mesh_vertices=self.current_centered_mesh_vertices,
                 current_quats=self.get_current_quat_and_pos(npify=True)[1],
                 current_pos=self.get_current_quat_and_pos(npify=True)[0],
                 stl_paths_for_sample=self.stl_paths_for_sample,
                 # tuple_id=self.tuple_id,
                 volumes=pb_util.calculate_object_volumes(self.pybullet_object_ids),
                 object_names=self.object_names,
                 moved_from_target=np.array(False),
                 future_moved_from_target=np.array(False),
                 sampled_path_idxs=np.array(self.sampled_path_idxs)
                 )

        if self.gen_table_pointcloud:
            o['table_pointcloud'] = camera_util.get_joint_pointcloud(self.cameras,
                                                                 obj_id=self.table_id,
                                                                 filter_table_height=False,
                                                                 return_ims=False)
        # if self.pointcloud_state:
        #     pointclouds = []
        #     # get a list of pointclouds for each object in the scene
        #     for pb_oid in self.pybullet_object_ids:
        #         pointcloud = camera_util.get_joint_pointcloud(self.cameras,
        #                                                       obj_id=pb_oid,
        #                                                       filter_table_height=False)
        #         pointclouds.append(pointcloud)
        #     o['rotated_raw_pointcloud'] = pointclouds
        if self.pointcloud_state:
            pointclouds = []
            uv_one_in_cams = []
            depths_arr = []
            # get a list of pointclouds for each object in the scene
            for current_oid in self.pybullet_object_ids:
                # pointcloud = camera_util.get_joint_pointcloud(self.cameras,
                #                                               obj_id=pb_oid,
                #                                               filter_table_height=False)

                pointcloud, uv_one_in_cam, depths = camera_util.get_joint_pointcloud(self.cameras,
                                                                                     obj_id=current_oid,
                                                                                     filter_table_height=False,
                                                                                     return_ims=False,
                                                                                     return_uv_cam_only=True)

                pointclouds.append(pointcloud)
                uv_one_in_cams.append(uv_one_in_cam)
                depths_arr.append(depths)

            o['rotated_raw_pointcloud'] = pointclouds
            o['uv_one_in_cam'] = uv_one_in_cams
            o['depths'] = depths_arr
        return o

    def render(self, scale_factor=1/2):
        return camera_util.get_image(cam_pkl=self.cam_pkl, scale_factor=scale_factor)

    def get_target_pos(self, current_object_id_flat):
        """
        Get a position that is above the tower. Place it there, and let forward simulation run.
        :param current_object_id_flat:
        :param object_ids:
        :return:
        """
        object_height, list_of_aabbs = get_object_height(self.pybullet_object_ids[current_object_id_flat],
                                                         ret_aabbs=True)

        # calculate target pos
        target_pos = np.zeros(3)

        # set z only
        max_height_obj_z, max_height_obj_id = get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited],
                                                     keep_table_height=True, scale=False, ret_max_height_id=True)

        target_pos[-1] = max_height_obj_z + object_height/2 + self.placement_offset

        max_height_obj_pos, _ = p.getBasePositionAndOrientation(max_height_obj_id)

        # replace this with COM of the existing tower
        # calculate COM positions
        # calculate mass of each object
        # take weighted average

        # target_pos[:2] = max_height_obj_pos[:2]

        fallen_objects = [self.pybullet_object_ids[idx] for idx in self.visited]

        if len(self.object_id_move_order) > 0:
            fallen_objects.remove(self.object_id_move_order[0])

        # if an object is not touching the table, remove it
        cp = p.getContactPoints()
        bodies_found_on_table = [tup[2] for tup in cp if (tup[1] == self.table_id and tup[1] != self.plane_id)]

        # print("ln480 cp")
        # print(cp)
        # print(bodies_found_on_table)
        for obj in fallen_objects:
            if obj in bodies_found_on_table:
                pass
            else:
                fallen_objects.remove(obj)

        print(fallen_objects)
        active_collection = [self.pybullet_object_ids[idx] for idx in self.visited]

        for obj in active_collection:
            if obj in fallen_objects:
                active_collection.remove(obj)

        print(active_collection)
        # tower_com = get_collection_com(active_collection)
        tower_com = max_height_obj_pos
        # tower_com = np.zeros(3)
        target_pos[:2] = tower_com[:2]
        return target_pos, list_of_aabbs

    def step(self, action: dict, debug_rotation_angle=False,
             get_pic=False, debug_draw_abb=False, falling_reset=True):
        """

        :param action: Dictionary of actions
            relative_quat: a quaternion that takes us from the current orientation in the world,
            to a new target orientation where the object is upright and stable
        :param debug_rotation_angle: Visualize rotation angle changing
        :return:
            state: rotated mesh pointcloud. [nO, num_points, 3]
            reward:
                number of objects which are in height maximizing and stacked position,
                starting from the object with lowest z-xis
            done:
                if we make one incorrect prediction, we are done...
        """

        print("\n\n")
        print(f"ln350 Starting timestep: {self.timestep[0]}")
        if debug_rotation_angle:
            pb_util.pb_key_loop("n")

        prev_sim_state = p.saveState()
        # TODO: do this correctly
        # Actual placement begins
        if "selected_object_id" in action.keys() and action['selected_object_id'] >=0:
            if self.pointcloud_state:
                assert "relative_quat" in action.keys()
            print("ln271 Actuating object...")
            current_object_id_flat = action['selected_object_id']
            # assert not float(current_object_id_flat) in self.visited, "Trying to visit already visited state"
            assert current_object_id_flat >=0, current_object_id_flat
            pybullet_oid = self.pybullet_object_ids[current_object_id_flat]
            assert pybullet_oid > 1 # should not be table or 0th thing

            if self.xy_noise_level:
                target_x_pos = np.random.normal(scale=self.xy_noise_level)
                target_y_pos = np.random.normal(scale=self.xy_noise_level)
            else:
                target_x_pos = 0.
                target_y_pos = 0.


            if self.pointcloud_state:
                pos, ori_quat = p.getBasePositionAndOrientation(pybullet_oid)

                if self.arm is not None:
                    self.arm.set_ee_pose(pos=pos, ori=ori_quat)

                target_quat = (R.from_quat(action['relative_quat']) * R.from_quat(ori_quat)).as_quat()

                p.resetBasePositionAndOrientation(pybullet_oid,
                                                  # self.sample_dict['target_pos'][current_object_id_flat],
                                                  [target_x_pos, target_y_pos, get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited],
                                                                                     keep_table_height=True, scale=False) + 2*get_object_height(pybullet_oid)],
                                                  target_quat)

            else:
                # object height must be calculated after the orientation is set... otherwise it will be wrong
                p.resetBasePositionAndOrientation(pybullet_oid,
                                                  # self.sample_dict['target_pos'][current_object_id_flat],
                                                  [target_x_pos, target_y_pos, get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited],
                                                                                     keep_table_height=True, scale=False) + 2*get_object_height(pybullet_oid)],
                                                  [0.,0.,0.,1.])

            if debug_rotation_angle:
                # print(f"Visualizing applied angle {action_angle}")
                pb_util.pb_key_loop("n")

            target_pos, list_of_aabbs = self.get_target_pos(current_object_id_flat
                                                            )
            # th_debug_id = p.addUserDebugText("TH", np.array([0., 0., get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited],
            #                                                      keep_table_height=True, scale=False)]))
            # tp_debug_id = p.addUserDebugText("TP", target_pos)

            if self.xy_noise_level:
                target_pos[0] = target_x_pos
                target_pos[1] = target_y_pos

            if debug_draw_abb:
                list_of_lines = draw_aabb_list(list_of_aabbs)

            if self.pointcloud_state:
                # the current ori is already correct because it was set prior
                pos, ori_quat = p.getBasePositionAndOrientation(pybullet_oid)

                # target_quat = (R.from_quat(action['relative_quat']) * R.from_quat(ori_quat)).as_quat()

                # loop over target pos until we touch something, then back up, and
                _, max_height_obj = get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited], ret_max_height_id=True)

                # step sim so contacts get updated
                p.stepSimulation()

                offset = 0

                new_tp = target_pos
                # p.setCollisionFilterPair(pybullet_oid, max_height_obj, -1, -1, 0)
                count = 0
                while not (bool(p.getContactPoints(pybullet_oid, max_height_obj)) or
                           bool(p.getContactPoints(pybullet_oid, self.table_id))) and count < 300000:
                    count += 1
                    print("ln460 going down")
                    print(target_pos + offset)
                    print(p.getContactPoints(pybullet_oid, max_height_obj))
                    offset -= .000001
                    new_tp = new_tp.copy()
                    new_tp[-1] = new_tp[-1] + offset
                    p.resetBasePositionAndOrientation(pybullet_oid,
                                                      new_tp,
                                                       target_quat)
                    p.stepSimulation()

                new_tp[-1] = new_tp[-1] - offset

                # target_quat = [0., 0., 0., 1.]
                p.resetBasePositionAndOrientation(pybullet_oid,
                                                  new_tp,
                                                  target_quat)
            else:
                # Drop at target pos, with certain quat
                p.resetBasePositionAndOrientation(pybullet_oid,
                                                  target_pos,
                                                  [0.,0.,0.,1.])
        else:
            print("ln302 Noop")
            # should be -1
            current_object_id_flat = action['selected_object_id']

        # after putting it right above the tower, take a nap
        # time.sleep(5)
        pb_util.pb_key_loop("m")

        if debug_rotation_angle:
            print("219")
            pb_util.pb_key_loop("n")

        if self.forward_sim_steps:
            for _ in range(self.forward_sim_steps):
                # print("ln166 forward simulating...")
                p.stepSimulation()

        if "selected_object_id" in action.keys() and action['selected_object_id'] >=0 and falling_reset:
            print("ln313 Falling reset!")
            # if the object moved over a certain threshold after placing it, the tower is falling...
            current_pos, current_ori = p.getBasePositionAndOrientation(pybullet_oid)

            self.object_id_move_order.append(pybullet_oid)
            if np.linalg.norm(current_pos - target_pos) > .05:
                moved_from_target=True
                self.future_moved_from_target = True
                # if the tower is falling... reset
                p.restoreState(prev_sim_state)

                visited_boolean_masks = convert_ints_to_boolean_mask(list(self.visited), self.num_objects)
                visited_indices = np.where(visited_boolean_masks == 1)[0]
                zipped = zip(visited_indices, [self.current_pos[idx][-1] for idx in visited_indices])

                visited_python_indices_sorted_by_height = [zip[0] for zip in sorted(zipped, key=lambda tup: tup[1])]

                for oid, poid in zip(visited_python_indices_sorted_by_height, [self.pybullet_object_ids[python_idx]
                                                                               for python_idx in visited_python_indices_sorted_by_height]):
                    p.resetBasePositionAndOrientation(poid, self.current_pos[oid], self.current_quats[oid])

                    for i in range(self.forward_sim_steps):
                        p.stepSimulation()
                current_object_id_flat = -1

            else:
                moved_from_target=False
        else:
            moved_from_target=False

        if debug_rotation_angle:
            print(f"ln238 Visualizing after forward simulation")
            pb_util.pb_key_loop("n")

        if "list_of_lines" in vars():
            remove_list_of_lines(list_of_lines)
        # Take the normals at the starting pose, and rotate them by the target quats
        target_quats = []
        for obj_id in self.pybullet_object_ids:
            pos, quat = p.getBasePositionAndOrientation(obj_id)
            target_quats.append(quat)

        target_quats_np = np.array(target_quats)

        out = mesh_util.update_tricenters_and_normals(self.object_names,
                                                      self.canonical_mesh_objects,
                                                      target_quats_np,
                                                      npify=True)
        self.current_tri_centers, self.current_phace_normals = out[-3], out[-2]

        reward = get_tower_height(self.pybullet_object_ids)
        done = False

        self.timestep = self.timestep + 1

        if get_pic:
            pic = camera_util.get_image(str(config.BANDU_ROOT / "pybullet/cams/left_cam1.pkl"), scale_factor=1)
            info['pic'] = pic

        if current_object_id_flat >=0:
            self.visited.add(int(current_object_id_flat))
        # print(f"ln284 visited: {self.visited}")
        if self.extra_config[current_object_id_flat]['block_type'] == "cap":
            self.capped = True
        self.current_pos, self.current_quats = self.get_current_quat_and_pos(npify=True)

        assert len(self.visited) <= self.num_objects, (self.visited, self.num_objects)

        # print("ln382 ccmv")
        # print(self.current_centered_mesh_vertices)
        # assert self.current_centered_mesh_vertices.dtype != np.dtype('O')

        o = dict(
                    pybullet_object_ids=np.array(self.pybullet_object_ids),
                    extra_config=self.extra_config,
                    current_quats=self.get_current_quat_and_pos(npify=True)[1],
                    current_pos=self.get_current_quat_and_pos(npify=True)[0],
                    stl_paths_for_sample=self.stl_paths_for_sample,
                    # tuple_id=self.tuple_id,
                    volumes=pb_util.calculate_object_volumes(self.pybullet_object_ids),
                    object_names=self.object_names,
                    moved_from_target=np.array(moved_from_target),
                    future_moved_from_target=np.array(self.future_moved_from_target),
                    object_id_move_order=self.object_id_move_order,
                    capped=np.asarray(self.capped),
                    visited=convert_ints_to_boolean_mask(list(self.visited), self.num_objects),
                    current_tri_centers=self.current_tri_centers,
                    current_phace_normals=self.current_phace_normals,
                    current_centered_mesh_vertices=self.current_centered_mesh_vertices,
                    timestep=self.timestep,
                    sampled_path_idxs=np.array(self.sampled_path_idxs)
                )

        if self.pointcloud_state:
            pointclouds = []
            uv_one_in_cams = []
            depths_arr = []
            # get a list of pointclouds for each object in the scene
            for current_oid in self.pybullet_object_ids:
                # pointcloud = camera_util.get_joint_pointcloud(self.cameras,
                #                                               obj_id=pb_oid,
                #                                               filter_table_height=False)
                pointcloud, uv_one_in_cam, depths = camera_util.get_joint_pointcloud(self.cameras,
                                                                                     obj_id=current_oid,
                                                                                     filter_table_height=False,
                                                                                     return_ims=True)

                pointclouds.append(pointcloud)
                uv_one_in_cams.append(uv_one_in_cam)
                depths_arr.append(depths)

            o['rotated_raw_pointcloud'] = pointclouds
            o['uv_one_in_cam'] = uv_one_in_cams
            o['depths'] = depths_arr
            if self.gen_table_pointcloud:
                o['table_pointcloud'] = camera_util.get_joint_pointcloud(self.cameras,
                                                                     obj_id=self.table_id,
                                                                     filter_table_height=False,
                                                                     return_ims=False)

        info = dict(num_stacked=np.array(bandu_util.get_number_stacked_objects(self.table_id, self.plane_id, self.num_objects)),
                    tower_height=get_tower_height([self.pybullet_object_ids[idx] for idx in self.visited],
                                                  keep_table_height=True, scale=False),
                    selected_object_id=action['selected_object_id'])
        return o, reward, done, info