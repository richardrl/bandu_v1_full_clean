import open3d.visualization
from torch.utils.data import Dataset
import pandas as pd
from utils import vis_util, mesh_util, surface_util, pointcloud_util, color_util
import open3d as o3d

# from utils.pointcloud_util import *
import numpy as np
from utils import camera_util

from bandu.config import BANDU_ROOT
import pickle
from scipy.spatial.transform import Rotation as R
import torch
import copy
import os
from pytorch3d import ops as pytorch3d_ops


def get_bti(batched_pointcloud,
            threshold_bottom_of_upper_region,
            threshold_top_of_bottom_region=None,
            max_z=None, min_z=None):
    """

    Gets bottom thresholded indicators. In other words, a binary map of the contact region.

    :param batched_pointcloud: np.ndarray[num_points, 3]
    :param threshold_bottom_of_upper_region: Bottom of upper region, expressed as fraction of total object height
    :param threshold_top_of_bottom_region: Top of bottom region, expressed as fraction of total object height

    The feasible contact region is between the top of bottom region <-> bottom of upper region
    :param line_search: search along the canonical axis for the section with the most support
    :return:
    """

    if threshold_top_of_bottom_region is not None:
        assert threshold_top_of_bottom_region < threshold_bottom_of_upper_region, (threshold_top_of_bottom_region, threshold_bottom_of_upper_region)

    # returns boolean where 1s rotated_pc_mean BACKGROUND and 0 means surface

    # since pointcloud is in the canonical position, we chop off the bottom points
    if max_z is None and min_z is None:
        max_z = np.max(batched_pointcloud[..., -1])
        min_z = np.min(batched_pointcloud[..., -1])
        object_heights = (max_z - min_z)
    else:
        object_heights = max_z - min_z

    threshold_object_height = object_heights * threshold_bottom_of_upper_region
    threshold_world_bottom_of_upper_region = min_z + threshold_object_height
    # threshold_world_bottom_of_upper_region = min_z + threshold_bottom_of_upper_region

    # if line_search:
    #     # start from [0, threshold_bottom_of_upper_region]
    #     # add 1% each time to both start, stop frac
    #     # try to fit a plane
    #     # pick the plane with the closest normal
    #
    # else:
    # batch_size x num_objects x num_points
    # returns true for all points that are ABOVE the z-threshold

    # collect all points which are above the threshold region as the background points

    # print("ln63 threshold world bottom of upper region")
    # print(threshold_world_bottom_of_upper_region)
    bti = np.greater(batched_pointcloud[..., -1], np.expand_dims(threshold_world_bottom_of_upper_region, axis=-1))

    if threshold_top_of_bottom_region:
        threshold_world_top_of_bottom_region = min_z + object_heights * threshold_top_of_bottom_region
        bti_bottom_region = np.less(batched_pointcloud[..., -1], np.expand_dims(threshold_world_top_of_bottom_region, axis=-1))

        bti = bti + bti_bottom_region

    # print(f"Found contact points {np.sum(1-bti)}")
    bti = bti[..., None]

    # assert np.sum(bti) > 0 and np.sum(bti) < batched_pointcloud.shape[0], np.sum(bti)
    return bti


def get_bti_from_rotated(rotated_batched_pointcloud, orientation_quat, threshold_frac,
                         linear_search=False,
                         max_z=None, min_z=None,
                         max_frac_threshold=.1
                         ):
    """
    Gets a single bti
    :param rotated_batched_pointcloud:
    :param orientation_quat:
    :param threshold_frac: Fraction of height to use the threshold on
    :return:
    """
    assert len(rotated_batched_pointcloud.shape) == 2
    assert len(orientation_quat.shape) == 1

    # canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud.cpu().data.numpy())
    canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud)
    if linear_search:
        # do a linear search over 100 pairs for the section that gives best oriented normal, for the CANONICAL pointcloud,
        # in terms of cosine distance to the negative gravity vector
        # delineate [lower, upper] values from 0% to 10%

        # keep increasing the threshold until you get enough points

        def find_bottom_threshold_interval(threshold_frac_inner):
            """

            Args:
                threshold_frac_inner: Scalar. The fraction of the z-range to use for the points container.

            Returns:

            """
            found_btis = []
            found_rotmats_distance_to_identity = []

            for lower_start_frac in np.linspace(0, max_frac_threshold, num=100):
                found_bti = get_bti(canonical, threshold_frac_inner + lower_start_frac, lower_start_frac, max_z=max_z, min_z=min_z)

                try:
                    relative_rotmat, plane_model = surface_util.get_relative_rotation_from_hot_labels(torch.as_tensor(canonical),
                                                                                                      torch.as_tensor(found_bti.squeeze(-1)),
                                                                                                      min_z=min_z,
                                                                                                      max_z=max_z)
                    # TODO: fix this
                    # print("Successfully found rotmat")
                except Exception as e:
                    # print("Failed to fit rotmat")
                    # print(e)
                    continue
                # print(relative_rotmat)
                found_rotmats_distance_to_identity.append(np.linalg.norm(relative_rotmat - np.eye(3)))
                found_btis.append(found_bti)
            return found_rotmats_distance_to_identity, found_btis

        try:
            for threshold_frac_inner in [threshold_frac, 2*threshold_frac, 4*threshold_frac, 5*threshold_frac]:
                outer_found_rotmats_distance_to_identity, outer_found_btis = find_bottom_threshold_interval(threshold_frac_inner)

                if outer_found_rotmats_distance_to_identity:
                    closest_to_identity_id = np.argmin(outer_found_rotmats_distance_to_identity)
                    break

            assert np.sum(outer_found_btis[closest_to_identity_id]) > 0, np.sum(outer_found_btis[closest_to_identity_id])
            assert np.sum(outer_found_btis[closest_to_identity_id]) < outer_found_btis[closest_to_identity_id].shape[0], \
                np.sum(outer_found_btis[closest_to_identity_id])
            return outer_found_btis[closest_to_identity_id]
        except:
            return get_bti(canonical, threshold_frac, 0, min_z=min_z,
                           max_z=max_z)
    else:
        return get_bti(canonical, threshold_frac, 0, min_z=min_z,
                       max_z=max_z)

def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


def absolute_dir_paths(directory):
    path = os.path.abspath(directory)
    # if this only returns one item, check that the directory is the parent of all the object directories
    return [entry.path for entry in os.scandir(path) if entry.is_dir()]


# e.g. /home/richard/improbable/spinningup/out/canonical_pointclouds/bandu_val/v2_test
def read_data_dir(samples_dir):
    """
    Assumes folder structure has "samples" as a child folder
    :param data_working_dir:
    :return:
    """
    assert samples_dir is not None
    assert samples_dir[-1] != "/"

    # if samples_dir[-1] == "/":
    #     samples_dir = samples_dir[:-1]
    # object_dirs = absolute_dir_paths(Path(data_working_dir) / "samples")

    # if this line fails, check that samples_dir is correct
    object_dirs = absolute_dir_paths(samples_dir)

    # column_names = ["filepath", "date", "take", "posX", "posY", "posZ", "quatX", "quatY", "quatZ", "quatW"]
    column_names = ["file_path", "object_name", "sample_idx"]
    df = pd.DataFrame(columns=column_names)

    for object_dir_path in object_dirs:
        # for sample_pkl in absolute_file_paths(object_dir_path):
        sample_file_paths = absolute_file_paths(object_dir_path)

        if not sample_file_paths:
            continue

        object_name = os.path.basename(os.path.normpath(object_dir_path))
        object_names = [object_name for _ in range(len(sample_file_paths))]

        # print("ln49 sfp")
        # print(sample_file_paths)

        # todo: change naming, this is really just the name now
        sample_idxs = [os.path.basename(os.path.normpath(sfp)).split(".")[0] for sfp in sample_file_paths]
        sample_df = pd.DataFrame(zip(sample_file_paths, object_names, sample_idxs),
                                 columns=column_names)

        df = df.append(sample_df, ignore_index=True)

    return df


# pd.set_option("display.max_rows", None, "display.max_columns", None)

class PybulletPointcloudDataset(Dataset):
    def __init__(self,
                 data_dir,
                 scale_aug="xyz",
                 max_z_scale=2,
                 min_z_scale=.5,
                 max_shear=.5,
                 rot_mag_bound=2 * np.pi,
                 rot_aug="xyz",
                 shear_aug="xy",
                 use_normals=False,
                 threshold_frac=1/10,
                 stats_dic=None,
                 center_fps_pc=False,
                 linear_search=True,
                 max_frac_threshold=.1,
                 dont_make_btb=False,
                 randomize_z_canonical=False,
                 further_downsample_frac=None,
                 augment_extrinsics=False,
                 depth_noise_scale=1.,
                 extrinsics_noise_scale=1.0,
                 use_realsense_extrinsics=False):

        self.data_df = read_data_dir(data_dir)
        self.data_dir = data_dir
        self.scale_aug = scale_aug
        self.max_z_scale = max_z_scale
        self.min_z_scale = min_z_scale
        self.max_shear = max_shear
        self.rot_mag_bound = rot_mag_bound
        self.rot_aug = rot_aug
        self.shear_aug = shear_aug
        self.use_normals = use_normals
        self.threshold_frac = threshold_frac

        self.stats_dic = stats_dic

        # do not call the below option if the pointcloud is already centered
        self.center_fps_pc = center_fps_pc

        self.linear_search = linear_search
        self.max_frac_threshold = max_frac_threshold

        # btb: bottom thresholded boolean aka the binary contact mask
        self.dont_make_btb = dont_make_btb
        self.randomize_z_canonical = randomize_z_canonical
        self.further_downsample_frac = further_downsample_frac

        if use_realsense_extrinsics:
            self.cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                                     camera_forward_z_offset=.2,
                                                     intrinsics_matrix=np.array([[888, 0, 630],
                                                                                 [0, 888, 360],
                                                                                 [0, 0, 1]], dtype=np.float64))
        else:
            self.cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                                     camera_forward_z_offset=.2)

        self.augment_extrinsics = augment_extrinsics
        self.depth_noise_scale = depth_noise_scale

        self.extrinsics_noise_scale = extrinsics_noise_scale

    def __len__(self):
        # returns number of rows in dataframe
        index = self.data_df.index
        return len(index)

    def __getitem__(self, item):
        # load from filepath
        df_row = self.data_df.loc[item]
        fp = df_row['file_path']
        main_dict = torch.load(fp)

        # sample uniformly to get fixed size
        # try:
        #     sampled_idxs = np.random.choice(np.arange(main_dict['aggregate_uv1incam_depth_and_cam_idxs'].shape[0]),
        #                                 size=2048, replace=False)
        # except:
        #     sampled_idxs = np.random.choice(np.arange(main_dict['aggregate_uv1incam_depth_and_cam_idxs'].shape[0]),
        #                                 size=2048, replace=True)

        # -> 2048, 4 where the last dimension is the cam idx
        # uniform_sampled_agg_depth_cam_idxs = main_dict['aggregate_uv1incam_depth_and_cam_idxs'][sampled_idxs, :]
        # partial_pcs = camera_util.convert_uv_depth_matrix_to_pointcloud(uniform_sampled_agg_depth_cam_idxs,
        #                                                   self.cameras)

        full_pc = np.concatenate(camera_util.convert_uv_depth_matrix_to_pointcloud(main_dict['aggregate_uv1incam_depth_and_cam_idxs'],
                                                          self.cameras), axis=0)
        selected_points, selected_indices = pytorch3d_ops.sample_farthest_points(torch.from_numpy(full_pc).unsqueeze(0), K=2048, random_start_point=True)

        partial_pcs = camera_util.convert_uv_depth_matrix_to_pointcloud(main_dict['aggregate_uv1incam_depth_and_cam_idxs'][selected_indices.squeeze(0), :],
                                                          self.cameras)

        pc = np.concatenate(partial_pcs, axis=0)

        # center pc
        pc = pc - main_dict['position']

        """
        Start Unit Test
        """
        # pcd = vis_util.make_point_cloud_o3d(R.from_quat(main_dict['rotated_quat']).inv().apply(pc),
        #                                     [1., 0., 0.])
        # # visualize
        # o3d.visualization.draw_geometries([pcd,
        #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0])])
        """
        End Unit Test
        """

        # augmentations and generations
        M = np.eye(3)

        # TODO: all these transforms are calculated based off the original PC, is that correct...
        # aug 1: scale
        if self.scale_aug == "xyz":
            # fps_before = fps_pc.copy()
            _, M_scale = pointcloud_util.scale_aug_pointcloud(pc,
                                            np.array(main_dict['rotated_quat']),
                                            self.max_z_scale, self.min_z_scale)

            M = M_scale @ M
        else:
            M_scale = np.eye(3)


        # aug 2: shear
        if self.shear_aug == "xy":
            _, M_tmp = pointcloud_util.shear_aug_pointcloud(pc,
                                            np.array(main_dict['rotated_quat']),
                                            self.max_shear)
            M = M_tmp @ M

        """
        Generate object-level augs
        """
        if self.rot_aug == "z":
            aug_rot = R.from_euler("z", np.random.uniform(self.rot_mag_bound))

            # save rotation
            resultant_quat = (aug_rot * R.from_quat(np.array(main_dict['rotated_quat']))).as_quat()
            # main_dict['rotated_quat'] = resultant_quat
        elif self.rot_aug == "xyz":
            aug_rot = R.random()
            resultant_quat = (aug_rot * R.from_quat(np.array(main_dict['rotated_quat']))).as_quat()
            # main_dict['rotated_quat'] = resultant_quat
        else:
            assert self.rot_aug is None
            # rotated_pc_placeholder[sample_idx, 0] = fps_pc
            resultant_quat = np.array(main_dict['rotated_quat'])

        working_partial_pcs = []
        working_partial_pcs_colors = []
        test_partial_pcs = []

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

        for partial_pc_idx, partial_pc in enumerate(partial_pcs):
            if partial_pc.size == 0:
                # sometimes one or more of the depth cameras won't capture the object
                # in that case we skip
                continue
            # center partial pc
            partial_pc = partial_pc - main_dict['position']

            # canonicalize before applying M, the shape transform
            canonical_partial_pc = R.from_quat(main_dict['rotated_quat']).inv().apply(partial_pc).copy()
            test_partial_pcs.append(canonical_partial_pc)

            # apply M
            # order: scale, shear aug ->
            # canonical_trans: all augs except final aug rotation applied
            # the canonical trans is the object in the stacked pose
            canonical_partial_transformed = (M @ canonical_partial_pc.T).T


            # -> ORIGINAL QUAT -> AUG QUAT AROUND Z
            partial_pc = copy.deepcopy(R.from_quat(resultant_quat).apply(canonical_partial_transformed))


            # aug 3: rot
            # NOTE: THIS MUST HAPPEN AFTER APPLYING THE OTHER AUGS, AND THE ORIGINAL ROTATION!!
            # partial_pc = aug_rot.apply(partial_pc)

            # aug 4: extrinsic trans in world frame
            if self.augment_extrinsics:
                partial_pc += (np.random.uniform(3) - .5) * np.array([.015, .015, .005]) * self.extrinsics_noise_scale

            # aug 5: augment with depth-conditional noise
            # convert to depth
            # augment with depth conditional noise
            # convert back to pc
            pc_in_cam = (np.linalg.inv(self.cameras[partial_pc_idx].cam_ext_mat) @
                        np.concatenate([partial_pc, np.ones((partial_pc.shape[0], 1))], axis=-1).T).T

            if pc_in_cam.size == 0:
                import pdb
                pdb.set_trace()

            assert pc_in_cam.size != 0, print(item)
            pc_in_cam[:, -2] = pointcloud_util.augment_depth_realsense(pc_in_cam[:, -2],
                                                                       coefficient_scale=self.depth_noise_scale).copy()

            partial_pc = (self.cameras[partial_pc_idx].cam_ext_mat @
                        pc_in_cam.T).T

            working_partial_pcs.append(partial_pc)

            working_partial_pcs_colors.append(np.tile(colors[partial_pc_idx], (partial_pc.shape[0], 1)))

        if self.use_normals:
            # do same operation on base pc
            # transform base pc, and index to get the new normals
            # canonical = R.from_quat(quat).inv().apply(base_pc)
            # base_pc_transformed = (M @ canonical.T).T
            # base_pc_transformed = R.from_quat(quat).apply(base_pc_transformed)

            # calculate new normals
            # TODO: update these comments
            # currently, we take the augmented pc, fit the normals -> simplest way
            pcd = vis_util.make_point_cloud_o3d(farthest_point_sampled_pointcloud, [1., 0., 0.])

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

            pcd.normals = o3d.utility.Vector3dVector(mesh_util.fix_normal_orientation(np.array(pcd.points),
                                                                                      np.array(pcd.normals), np.zeros(3)))

            fps_normals_transformed = np.array(pcd.normals)

            pcd = vis_util.make_point_cloud_o3d(farthest_point_sampled_pointcloud, [1., 0., 0.])
            pcd.normals = o3d.utility.Vector3dVector(mesh_util.fix_normal_orientation(np.array(pcd.points),
                                                                                      fps_normals_transformed,
                                                                                      np.zeros(3)))
            # visualize
            # o3d.visualization.draw_geometries([pcd])

            # base_normals_transformed = np.array(pcd.normals)

            # fps_normals_transformed = base_normals_transformed[fps_indices]
        else:
            fps_normals_transformed = None

        # add object dimension for solo object
        # print("ln422 pc")
        # print(pc)

        pc = np.concatenate(working_partial_pcs, axis=0)[:, :3]
        main_dict['rotated_pointcloud'] = np.expand_dims(pc, axis=0).astype(float)

        # it is important we set rotated_quat here. rotated_quat is assumed to be just the object pose
        # (before augmentation) for most fo the above code.
        main_dict['rotated_quat'] = resultant_quat

        # with pd.option_context('display.max_rows', None,
        #                        'display.max_columns', None,
        #                        'display.precision', 3,
        #                        ):
        #     print(df_row)
        #     print(df_row['file_path'])

        if not self.dont_make_btb:
            # make bti on fully augmented and noised pc
            # import pdb
            # pdb.set_trace()
            main_dict['bottom_thresholded_boolean'] = get_bti_from_rotated(np.concatenate(working_partial_pcs, axis=0)[:, :3],
                                            resultant_quat, self.threshold_frac, self.linear_search,
                                                                           max_z=main_dict['canonical_max_height']*M_scale[2, 2],
                                                                           min_z=main_dict['canonical_min_height']*M_scale[2, 2],
                                                                           max_frac_threshold=self.max_frac_threshold).astype(float).squeeze(-1)

            """
            Start Unit Test
            """
            # print("ln427")
            # canonical_pc = np.concatenate(test_partial_pcs, axis=0)[:, :3]

            pc_colors = np.concatenate(working_partial_pcs_colors, axis=0)

            # pcd = vis_util.make_point_cloud_o3d(canonical_pc, pc_colors)
            # o3d.visualization.draw_geometries([pcd,
            #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0])
            #                                    ])



            """
            End Unit Test
            """

            """
            Start Visualize Contact Points
            """

            # working_pc = np.concatenate(working_partial_pcs, axis=0)[:, :3]
            #
            # o3d.visualization.draw_geometries([
            #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0]),
            #                                    vis_util.make_point_cloud_o3d(
            #                                        working_pc,
            #                                        color=vis_util.make_colors(main_dict['bottom_thresholded_boolean'],
            #                                                          background_color=color_util.MURKY_GREEN,
            #                                                          surface_color=color_util.RED))
            #                                    ])

            """
            End Visualize Contact Points
            """
            # if np.sum(1-main_dict['bottom_thresholded_boolean']) < 15:
            #
            #
            #     vis_util.make_point_cloud_o3d(pc,
            #                                   color=vis_util.make_colors(main_dict['bottom_thresholded_boolean'],
            #                                                     background_color=color_util.MURKY_GREEN,
            #                                                     surface_color=color_util.YELLOW))
            #
            #     import pdb
            #     pdb.set_trace()
            #     main_dict['bottom_thresholded_boolean'] = get_bti_from_rotated(pc,
            #                                                                    resultant_quat, self.threshold_frac,
            #                                                                    self.linear_search,
            #                                                                    max_z=main_dict['canonical_max_height'] *
            #                                                                          M_scale[2, 2],
            #                                                                    min_z=main_dict['canonical_min_height'] *
            #                                                                          M_scale[2, 2],
            #                                                                    max_frac_threshold=self.max_frac_threshold).astype(
            #         float).squeeze(-1)

            # assert np.sum(1-main_dict['bottom_thresholded_boolean']) >= 15, print(np.sum(1-main_dict['bottom_thresholded_boolean']))

        # 1-btb because 0s are contact points, 1s are background points
        if self.randomize_z_canonical:
            canonical_quat = R.from_euler("z", np.random.uniform(0, 2*np.pi)).as_quat()
            main_dict['canonical_quat'] = canonical_quat

            # undo the quat to get into canonical position, then apply the canonical rotation
            main_dict['relative_quat'] = (R.from_quat(canonical_quat) * R.from_quat(resultant_quat).inv()).as_quat()
            main_dict['canonical_pointcloud'] = R.from_quat(main_dict['relative_quat']).apply(pc)
        else:
            main_dict['relative_quat'] = R.from_quat(resultant_quat).inv().as_quat()
            main_dict['canonical_pointcloud'] = R.from_quat(main_dict['relative_quat']).apply(pc)

        if self.stats_dic:
            main_dict.update({k: np.expand_dims(np.array(v), 0) for k, v in self.stats_dic.items()})


        del main_dict['aggregate_uv1incam_depth_and_cam_idxs']
        return main_dict


if __name__ == '__main__':
    pcdset = PybulletPointcloudDataset("../out/datasets/bandu_train/jan18_train/voxelized_samples",
                                       augment_extrinsics=True,
                                       depth_noise_scale=1,
                                       threshold_frac=.02,
                                       extrinsics_noise_scale=.5,
                                       max_frac_threshold=.06,
                                       use_realsense_extrinsics=False)
    # todo: still some samples have no contact points
    # augment_extrinsics = True
    # depth_noise_scale = 1.5
    sample = pcdset.__getitem__(np.random.randint(len(pcdset)))