import torch
import numpy as np
import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import sys
import pandas as pd
from utils import vis_util, mesh_util
import open3d as o3d
from data_generation.sim_dataset import get_bti, get_bti_from_rotated
from utils.pointcloud_util import *


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


def absolute_dir_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_dir()]


# e.g. /home/richard/improbable/spinningup/out/canonical_pointclouds/bandu_val/v2_test
def read_data_dir(samples_dir):
    """
    Assumes folder structure has "samples" as a child folder
    :param data_working_dir:
    :return:
    """
    assert samples_dir[-1] != "/"
    # object_dirs = absolute_dir_paths(Path(data_working_dir) / "samples")
    object_dirs = absolute_dir_paths(samples_dir)

    # column_names = ["filepath", "date", "take", "posX", "posY", "posZ", "quatX", "quatY", "quatZ", "quatW"]
    column_names = ["file_path", "object_name", "sample_idx"]
    df = pd.DataFrame(columns=column_names)

    for object_dir_path in object_dirs:
        # for sample_pkl in absolute_file_paths(object_dir_path):
        sample_file_paths = absolute_file_paths(object_dir_path)
        object_name = os.path.basename(os.path.normpath(object_dir_path))
        object_names = [object_name for _ in range(len(sample_file_paths))]

        print("ln49 sfp")
        # print(sample_file_paths)
        sample_idxs = [int(os.path.basename(os.path.normpath(sfp)).split(".")[0]) for sfp in sample_file_paths]
        sample_df = pd.DataFrame(zip(sample_file_paths, object_names, sample_idxs),
                                 columns=column_names)

        df = df.append(sample_df, ignore_index=True)

    return df


# pd.set_option("display.max_rows", None, "display.max_columns", None)

class PointcloudDataset(Dataset):
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
                 further_downsample_frac=None):
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
        self.center_fps_pc = center_fps_pc

        self.linear_search = linear_search
        self.max_frac_threshold = max_frac_threshold
        self.dont_make_btb = dont_make_btb
        self.randomize_z_canonical = randomize_z_canonical
        self.further_downsample_frac = further_downsample_frac

    def __len__(self):
        # returns number of rows in dataframe
        index = self.data_df.index
        return len(index)

    def __getitem__(self, item):
        # load from filepath
        df_row = self.data_df.loc[item]
        fp = df_row['file_path']
        main_dict = torch.load(fp)


        # fps_pc = get_farthest_point_sampled_pointcloud(main_dict['rotated_pointcloud'],
        #                                                                         2048)

        farthest_point_sampled_pointcloud = main_dict['rotated_pointcloud']
        if self.further_downsample_frac:
            farthest_point_sampled_pointcloud = farthest_point_sampled_pointcloud[np.random.choice(farthest_point_sampled_pointcloud.shape[0], 512, replace=False)]
        if self.center_fps_pc:
            farthest_point_sampled_pointcloud = farthest_point_sampled_pointcloud - main_dict['position']

        # augmentations and generations
        M = np.eye(3)

        # aug 1: scale
        if self.scale_aug == "xyz":
            # fps_before = fps_pc.copy()
            _, M_scale = scale_aug_pointcloud(farthest_point_sampled_pointcloud,
                                            main_dict['rotated_quat'],
                                            self.max_z_scale, self.min_z_scale)

            M = M_scale @ M
        else:
            M_scale = np.eye(3)


        # aug 2: shear
        if self.shear_aug == "xy":
            _, M_tmp = shear_aug_pointcloud(farthest_point_sampled_pointcloud,
                                            main_dict['rotated_quat'],
                                            self.max_shear)
            M = M_tmp @ M

        # finally, set the PC
        # canonicalize
        canonical = R.from_quat(main_dict['rotated_quat']).inv().apply(farthest_point_sampled_pointcloud)

        # apply M
        # order: scale, shear aug ->
        # canonical_trans: all augs except final aug rotation applied
        # the canonical trans is the object in the stacked pose
        canonical_trans = (M @ canonical.T).T

        # -> ORIGINAL QUAT -> AUG QUAT AROUND Z
        farthest_point_sampled_pointcloud = R.from_quat(main_dict['rotated_quat']).apply(canonical_trans)

        # aug 3: rot
        # NOTE: THIS MUST HAPPEN AFTER APPLYING THE OTHER AUGS, AND THE ORIGINAL ROTATION!!
        if self.rot_aug == "z":
            aug_rot = R.from_euler("z", np.random.uniform(self.rot_mag_bound))

            # -> augmented rotation applied to original rotated pc
            farthest_point_sampled_pointcloud = aug_rot.apply(farthest_point_sampled_pointcloud)

            # save rotation
            resultant_quat = (aug_rot * R.from_quat(np.array(main_dict['rotated_quat']))).as_quat()
            main_dict['rotated_quat'] = resultant_quat
            # rotated_quats[sample_idx, 0] = resultant_quat
        elif self.rot_aug == "xyz":
            aug_rot = R.random()
            farthest_point_sampled_pointcloud = aug_rot.apply(farthest_point_sampled_pointcloud)
            resultant_quat = (aug_rot * R.from_quat(np.array(main_dict['rotated_quat']))).as_quat()
            main_dict['rotated_quat'] = resultant_quat
        else:
            assert self.rot_aug is None
            # rotated_pc_placeholder[sample_idx, 0] = fps_pc
            resultant_quat = np.array(main_dict['rotated_quat'])
            # rotated_quats[sample_idx, 0] = resultant_quat

        # pcd = vis_util.make_point_cloud_o3d(fps_pc, [1., 0., 0.])
        # # visualize
        # o3d.visualization.draw_geometries([pcd])
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

        # return (before_aug_fps_pc, resultant_quat, fps_pc, fps_normals_transformed)

        # add object dimension for solo object
        main_dict['rotated_pointcloud'] = np.expand_dims(farthest_point_sampled_pointcloud, axis=0).astype(float)

        # with pd.option_context('display.max_rows', None,
        #                        'display.max_columns', None,
        #                        'display.precision', 3,
        #                        ):
        #     print(df_row)
        #     print(df_row['file_path'])

        if not self.dont_make_btb:
            main_dict['bottom_thresholded_boolean'] = get_bti_from_rotated(farthest_point_sampled_pointcloud,
                                            resultant_quat, self.threshold_frac, self.linear_search,
                                                                           max_z=main_dict['canonical_max_height']*M_scale[2, 2],
                                                                           min_z=main_dict['canonical_min_height']*M_scale[2, 2],
                                                                           max_frac_threshold=self.max_frac_threshold).astype(float).squeeze(-1)
            assert np.sum(1-main_dict['bottom_thresholded_boolean']) >= 15, print(np.sum(1-main_dict['bottom_thresholded_boolean']))

        # 1-btb because 0s are contact points, 1s are background points

        if self.randomize_z_canonical:
            canonical_quat = R.from_euler("z", np.random.uniform(0, 2*np.pi)).as_quat()
            main_dict['canonical_quat'] = canonical_quat

            # undo the quat to get into canonical position, then apply the canonical rotation
            main_dict['relative_quat'] = (R.from_quat(canonical_quat) * R.from_quat(resultant_quat).inv()).as_quat()
            main_dict['canonical_pointcloud'] = R.from_quat(main_dict['relative_quat']).apply(farthest_point_sampled_pointcloud)
        else:
            main_dict['relative_quat'] = R.from_quat(resultant_quat).inv().as_quat()
            main_dict['canonical_pointcloud'] = R.from_quat(main_dict['relative_quat']).apply(farthest_point_sampled_pointcloud)

        if self.stats_dic:
            main_dict.update({k: np.expand_dims(np.array(v), 0) for k, v in self.stats_dic.items()})
        return main_dict


if __name__ == '__main__':
    pcdset = PointcloudDataset("/home/richard/improbable/spinningup/out/canonical_pointclouds/bandu_val/v2_test/samples")
    sample = pcdset.__getitem__(0)
