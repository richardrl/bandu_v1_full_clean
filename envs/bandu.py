import gym.utils.seeding
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from bandu.utils import *
from bandu import config
from itertools import product
from pathlib import Path
from supervised_training.utils.pb_spinningup_util import get_object_height


class BanduSimpleEnv(gym.Env):
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
                 scale_factor=1.0
                 ):
        # super().__init__(*args, **kwargs)
        # kwargs = train_dataset_params
        # self.dataset = BanduHandspecifiedMultiobject(*args, **dataset_kwargs)
        scene_ids = bandu_util.load_scene(p_connect_type=p_connect_type, realtime=realtime)

        self.p_connect_type = p_connect_type
        self.cam_pkl = cam_pkl
        self.num_objects = num_objects
        self.reward_type = reward_type
        self.forward_sim_steps = forward_sim_steps
        assert reward_type in ["height", "binary"]
        self.first_reset_completed = False
        self.debug = debug
        self.start_rotation_type = start_rotation_type

        import glob
        if urdf_dir:
            assert isinstance(urdf_dir, str)
            assert isinstance(stl_dir, str)
            assert urdfs is None
            assert stls is None

            # scan urdf_path dir for the urdfs
            # scan stl dir for the stls
            urdfs = [f for f in glob.glob(str(Path(urdf_dir) / "**/*.urdf_path"), recursive=True)]
            urdfs += [f for f in glob.glob(str(Path(urdf_dir) / "**/*.URDF"), recursive=True)]
            stls = [f for f in glob.glob(str(Path(stl_dir) / "**/*.stl"), recursive=True)]
            stls += [f for f in glob.glob(str(Path(stl_dir) / "**/*.STL"), recursive=True)]
            assert len(urdfs) == len(stls), (len(urdfs), len(stls))
            assert len(urdfs) > 0

        self.urdf_combos = list(product(urdfs, repeat=num_objects))
        self.stl_combos = list(product(stls, repeat=num_objects))
        self.randomize_mesh_vertices = randomize_mesh_vertices
        self.scale_factor = scale_factor

    def get_nparr_of_current_quats(self):
        quats = []
        for obj_id in self.object_ids:
            pos, quat = p.getBasePositionAndOrientation(obj_id)
            quats.append(quat)
        return np.array(quats)

    def reset(self):
        if self.first_reset_completed:
            bandu_util.remove_bandu_objects(self.num_objects)
        else:
            self.first_reset_completed = True

        # sample_dict = self.dataset.gen_sample()

        # uniformly sample_idx
        tuple_id = np.random.randint(0, len(self.urdf_combos))
        urdf_paths_for_sample = self.urdf_combos[tuple_id]
        stl_paths_for_sample = self.stl_combos[tuple_id]

        self.object_names = bandu_util.get_object_names(urdf_paths_for_sample)

        self.object_ids = bandu_util.load_bandu_objects(urdf_paths=urdf_paths_for_sample,
                                                        scale_factor=self.scale_factor)

        self.canonical_mesh_objects = bandu_util.load_centered_meshes(stl_paths_for_sample)

        canonical_meshes_vertices = bandu_util.load_meshes_vertices(stl_paths_for_sample, self.randomize_mesh_vertices)


        for idx, oid in enumerate(self.object_ids):
            initial_orientation = bandu_util.get_initial_orientation(self.start_rotation_type)

            start_pos = data_util.get_position_for_drop_above_table(height_off_set=.2)
            p.resetBasePositionAndOrientation(oid, start_pos, initial_orientation.as_quat())

        mesh_vertices_padded = np.stack(misc_util.pad_same_size(canonical_meshes_vertices), axis=0)

        def update_meshes(canonical_mesh_vertices, quats):
            out = []
            for obj_id in range(len(self.object_ids)):
                out.append(R.from_quat(quats[obj_id]).apply(canonical_mesh_vertices[obj_id]))
            return np.array(out)

        # self.current_centered_mesh_vertices = R.from_quat(self.get_nparr_of_current_quats()).apply(mesh_vertices_padded)

        self.current_centered_meshes_vertices = update_meshes(canonical_meshes_vertices, self.get_nparr_of_current_quats())


        mtc_arr, mfn_arr, labels_arr, start_mtc_arr, start_mfn_arr, positive_index_arr = \
            mesh_util.update_tricenters_and_normals(self.object_names,
                                                    self.canonical_mesh_objects,
                                                    self.get_nparr_of_current_quats(),
                                                    npify=False)

        self.current_phace_normals = start_mfn_arr
        self.current_tri_centers = start_mtc_arr

        self.mesh_phace_normals = mfn_arr
        self.mesh_tri_centers = mtc_arr
        self.num_objects = len(self.object_ids)

        self.current_object_id = np.array([0])
        self.canonical_mesh_normal_labels = labels_arr


        # nsmnil = np.expand_dims(np.argmax(self.canonical_mesh_normal_labels[0]), axis=0)

        o = dict(current_tri_centers=self.current_tri_centers,
                 current_phace_normals=self.current_phace_normals,
                 object_ids=self.object_ids,
                 current_object_id=self.current_object_id,
                 next_state_optimal_angle=0)
                 # next_selected_mesh_normals_label=nsmnil) # optimal angle is 0 for timestep 0
        return o

    def render(self, scale_factor=1/2):
        return camera_util.get_image(cam_pkl=self.cam_pkl, scale_factor=scale_factor)

    def step(self, action, debug_rotation_angle=False):
        """

        :param action: Object ID and quaternion
        :param debug_rotation_angle: Visualize rotation angle changing
        :return:
            state: rotated mesh pointcloud. [nO, num_points, 3]
            reward:
                number of objects which are in height maximizing and stacked position,
                starting from the object with lowest z-xis
            done:
                if we make one incorrect prediction, we are done...
        """
        # assert len(action) == 2, action
        # action is a surface normal
        # action_object_id = action[0] # NOTE: This is the PYTHON index, not PYBULLET INDEX!!!
        if debug_rotation_angle:
            print("ln86")
            pb_util.pb_key_loop("n")


        # TODO: do this correctly
        current_object_id_flat = np.asscalar(self.current_object_id)
        print("ln91 current_object_id_flat")
        print(current_object_id_flat)
        action_object_pybullet_id = self.object_ids[current_object_id_flat]
        assert action_object_pybullet_id > 1 # should not be table or 0th thing

        # action_surfaces_logits = action[1]

        # print("ln90")
        # print(action)
        if isinstance(action, dict):
            if 'normals_classifier0' in action.keys():
                action_surfaces_logits = action['normals_classifier0']
            else:
                action_surfaces_logits = None

            if "angle_regressor1" in action:
                action_angle = action['angle_regressor1']
                assert len(action_angle.shape) <= 2
        else:
            action_surfaces_logits = action

        if action_surfaces_logits is not None:
            assert len(self.current_phace_normals.shape) == 3
            # nO, num_normals, 3 -> num_normals, 3 -> batch_dim=1, 1, num_normals, 3
            assert np.isscalar(current_object_id_flat), current_object_id_flat

            # Get a relative quaternion from the object surface normals
            cpn = self.current_phace_normals[None, None, current_object_id_flat, ...]
            # assert cpn.shape[2] == 12, cpn.shape
            action_quat = data_util.get_rot_quats_from_surface_normals_and_predicted_indices(cpn,
                                                                                             np.expand_dims(action_surfaces_logits, axis=0)).squeeze()
            try:
                assert not np.any(np.isnan(action_quat)), action_quat
            except:
                import pdb
                pdb.set_trace()
            assert len(action_surfaces_logits.shape) ==1, action_surfaces_logits

        else:
            action_quat = R.from_euler("z", [0]).as_quat()

        current_pos, current_quat = p.getBasePositionAndOrientation(action_object_pybullet_id)



        if debug_rotation_angle and action_surfaces_logits is not None:
            print(f"ln139 action quat {action_quat}")
            print(f"ln139 current quat {current_quat}")
            p.resetBasePositionAndOrientation(action_object_pybullet_id,
                                              # self.sample_dict['target_pos'][current_object_id_flat],
                                              np.array([0, 0, config.TABLE_HEIGHT + .4]),
                                              (R.from_quat(action_quat) * R.from_quat(current_quat)).as_quat())
            p.stepSimulation()
            pb_util.pb_key_loop("n")

        # Optionally, apply the angle as well...
        if isinstance(action, dict) and "angle_regressor1" in action:
            # print(f"ln118 Using action angle {action_angle}")
            action_quat = (R.from_euler("z", [action_angle.squeeze()]) * R.from_quat(action_quat)).as_quat().squeeze()


        assert self.current_object_id < self.num_objects, self.current_object_id


        ### Update state
        # rotate the existing pointcloud
        nO, num_points, three_dim = self.current_centered_meshes_vertices.shape
        self.current_centered_meshes_vertices = R.from_quat(action_quat).apply(self.current_centered_meshes_vertices.reshape(nO * num_points, 3)).reshape(nO, num_points, 3)

        object_height, list_of_aabbs = get_object_height(self.object_ids[current_object_id_flat], ret_aabbs=True)

        target_pos = np.zeros(3)
        target_pos[-1] = bandu_util.get_tower_height(self.object_ids[:current_object_id_flat], keep_table_height=True, scale=False) \
                         + object_height/2

        # Drop at target pos, with certain quat
        p.resetBasePositionAndOrientation(action_object_pybullet_id,
                                          # self.sample_dict['target_pos'][current_object_id_flat],
                                          target_pos,
                                          (R.from_quat(action_quat.squeeze()) * R.from_quat(current_quat)).as_quat())

        if debug_rotation_angle:
            print((R.from_quat(action_quat.squeeze()) * R.from_quat(current_quat)).as_quat())
            pb_util.pb_key_loop("n")

        if self.forward_sim_steps:
            for _ in range(self.forward_sim_steps):
                # print("ln166 forward simulating...")
                p.stepSimulation()
        if debug_rotation_angle:
            print("167")
            print(f"Visualizing after forward simulation")
            pb_util.pb_key_loop("n")
        # Take the normals at the starting pose, and rotate them by the target quats
        target_quats = []
        for obj_id in self.object_ids:
            pos, quat = p.getBasePositionAndOrientation(obj_id)
            target_quats.append(quat)

        target_quats_np = np.array(target_quats)

        out = mesh_util.update_tricenters_and_normals(self.object_names,
                                                      self.canonical_mesh_objects,
                                                      target_quats_np,
                                                      npify=True)
        self.current_tri_centers, self.current_phace_normals = out[-3], out[-2]

        reward = bandu_util.get_tower_height(self.object_ids)
        done = False

        # if self.timestep < self.num_objects - 1:
        self.current_object_id = self.current_object_id + 1

        if debug_rotation_angle:
            print("ln207")
            print("Post movement debug")
            pb_util.pb_key_loop("n")

        if self.start_rotation_type == "planar":
            try:
                # Get previous object pb_id's rotation, and rotate by pi/2
                current_object_id = self.object_ids[current_object_id_flat]
                _, cur_quat = p.getBasePositionAndOrientation(current_object_id)

                cur_z_angle = R.from_quat(cur_quat).as_euler("XYZ")[-1]

                _, next_quat = p.getBasePositionAndOrientation(current_object_id + 1)

                next_obj_z_angle = R.from_quat(next_quat).as_euler("XYZ")[-1]
                print("ln223")
                print(cur_z_angle)
                next_target_z_angle = cur_z_angle + np.pi

                next_relative_z_angle = next_target_z_angle - next_obj_z_angle

                if next_relative_z_angle > np.pi * 2:
                    next_state_optimal_angle = next_relative_z_angle - 2*np.pi
                elif next_relative_z_angle < 0:
                    next_state_optimal_angle = next_relative_z_angle + 2*np.pi
                else:
                    next_state_optimal_angle = next_relative_z_angle
                print("ln233")
                print(next_state_optimal_angle)

                # between 0 and 2*np.pi -> between -np.pi and np.pi
                if next_state_optimal_angle > np.pi:
                    next_state_optimal_angle = -np.abs(next_state_optimal_angle - 2*np.pi)

            except:
                next_state_optimal_angle = 0

        # nsmnil = np.expand_dims(np.argmax(self.canonical_mesh_normal_labels[current_object_id_flat + 1]), axis=0) if current_object_id_flat < self.num_objects -1 else None
        return dict(current_tri_centers=self.current_tri_centers,
                    current_phace_normals=self.current_phace_normals,
                    object_ids=self.object_ids,
                    current_object_id=self.current_object_id), reward, done, {}
                    # next_state_optimal_angle=next_state_optimal_angle if self.start_rotation_type == "planar" else None,
                    # next_selected_mesh_normals_label=nsmnil), \
