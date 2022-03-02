# load original dataset
# voxelizes the samples and saves to disk
from data_generation.sim_dataset import PybulletPointcloudDataset
import torch
from utils import pointcloud_util, camera_util, vis_util
from pathlib import Path
import sys
from deco import *
import itertools
import time
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import os
import open3d
import time

# original_data_dir = "/home/richard/improbable/spinningup/out/canonical_pointclouds/bandu_val/test_v2/samples"
original_data_dir = sys.argv[1]
# num_fps_samples = int(sys.argv[2])
# randomize_noise = bool(int(sys.argv[3]))

pcdset_original = PybulletPointcloudDataset(original_data_dir)

# new_samples_dir = Path(original_data_dir) / ".." / f"fps_randomizenoise{randomize_noise}_numfps{num_fps_samples}_samples"
new_samples_dir = Path(original_data_dir) / ".." / f"voxelized_samples"


object_names = pcdset_original.data_df.object_name.unique()

for obj_name in object_names:
    fd_dir = new_samples_dir / obj_name

    os.umask(0)
    fd_dir.mkdir(parents=True, exist_ok=True)

cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                    camera_forward_z_offset=.2)


def create_aggregate_uv1incam_depth_and_cam_idxs(uv_one_in_cam,
                                                 depths,
                                                 voxel_downsample,
                                                 output_pcd=None,
                                                 cubic_id_to_original_indices=None):
    """

    Assume 0 size pointclouds have been filtered out.

    :param uv_one_in_cam: List[np.ndarray[3, num_points]]
    :param depths: List[np.ndarray[num_points]]
    :param voxel_downsample:
    :param output_pcd:
    :param cubic_id_to_original_indices:
    :return:
    """
    assert uv_one_in_cam[0].shape[0] == 3

    # segmented depth is the depth belonging to cam after undergoing voxelization
    cam_idx_to_segmented_depth = [[] for _ in range(len(cameras))]
    cam_idx_to_segmented_uv_one_in_cam = [[] for _ in range(len(cameras))]

    original_pc_idx_to_cam_idxs = []
    for cam_idx in range(len(uv_one_in_cam)):
        original_pc_idx_to_cam_idxs.append(np.ones(uv_one_in_cam[cam_idx].shape[-1], dtype=np.int32) * cam_idx)
    original_pc_idx_to_cam_idxs = np.concatenate(original_pc_idx_to_cam_idxs, axis=0)

    aggregate_uv_one_in_cam = np.concatenate(uv_one_in_cam, axis=-1)

    # -> total_points,
    aggregate_depths = np.concatenate(depths, axis=-1)

    # cam_idx_to_segmented_depth: List[np.ndarray[num_points]]
    # cam_idx_to_segmented_uv_one_in_cam: List[np.ndarray[3, num_points]]
    if voxel_downsample:
        for pt_idx in range(len(np.array(output_pcd.points))):
            # gets the mode index
            filtered_lst = list(filter(lambda x: x != -1, cubic_id_to_original_indices[pt_idx]))

            original_idx_corresponding_to_voxel_cubic_id = max(set(filtered_lst), key=filtered_lst.count)

            pt_cam_idx = original_pc_idx_to_cam_idxs[original_idx_corresponding_to_voxel_cubic_id]

            cam_idx_to_segmented_uv_one_in_cam[pt_cam_idx].append(
                aggregate_uv_one_in_cam[:, original_idx_corresponding_to_voxel_cubic_id])
            cam_idx_to_segmented_depth[pt_cam_idx].append(aggregate_depths[original_idx_corresponding_to_voxel_cubic_id])

        cam_idx_to_segmented_depth = [np.array(_) for _ in cam_idx_to_segmented_depth]
        cam_idx_to_segmented_uv_one_in_cam = [np.array(_) for _ in cam_idx_to_segmented_uv_one_in_cam]
    else:
        cam_idx_to_segmented_depth = [np.array(_) for _ in depths]
        cam_idx_to_segmented_uv_one_in_cam = [np.array(_) for _ in uv_one_in_cam]


    # repeat the procedure with uniform sampling to get a fixed pointcloud size
    # sample 2048 points
    # track the camera indices so we can get the segmented depth later
    # broadcast the camera idxs so we can use them for indexing
    cam_idxs_to_repeated_cam_idxs = [np.ones(cam_idx_to_segmented_depth[cam_idx].shape[0],
                                             dtype=np.int32) * cam_idx for cam_idx in range(len(cameras))]

    # make it all a big matrix for easy uniform sampling

    # -> num_points, 5

    # cam_idx_to_segmented_uv_one_in_cam: List[np.ndarray[num_points, 3]]
    # cam_idx_to_segmented_depth: List[np.ndarray[num_points, ]]

    # filter out all the zero point arrays

    cam_idx_to_segmented_uv_one_in_cam = [_ for _ in cam_idx_to_segmented_uv_one_in_cam if _ != []]
    cam_idx_to_segmented_depth = [_ for _ in cam_idx_to_segmented_depth if _ != []]
    cam_idxs_to_repeated_cam_idxs = [_ for _ in cam_idxs_to_repeated_cam_idxs if _ != []]

    try:
        aggregate_uv1incam_depth_and_cam_idxs = np.concatenate([
            np.concatenate(cam_idx_to_segmented_uv_one_in_cam, axis=0),
            np.expand_dims(np.concatenate(cam_idx_to_segmented_depth, axis=-1), axis=-1),
            np.expand_dims(np.concatenate(cam_idxs_to_repeated_cam_idxs, axis=-1), axis=-1),
        ], axis=-1)
    except:
        import pdb
        pdb.set_trace()
    return aggregate_uv1incam_depth_and_cam_idxs

import open3d as o3d
# @concurrent
def uvd_to_segmented_uvd(depths, uv_one_in_cam, row, dic, original_centered_pc):
    """

    :param depths: List[numpy.ndarray]
    :param uv_one_in_cam:
    :param row:
    :param dic:
    :param original_centered_pc:
    :return: Object-segmented depth
    """
    # center at COM
    # original_centered_pc = original_centered_pc - dic['position']

    # record camera index for each point
    # downsample according to XYZ, while recording indices

    pcd = vis_util.make_point_cloud_o3d(original_centered_pc, color=[0, 0, 0])

    obb = open3d.geometry.OrientedBoundingBox.create_from_points(open3d.utility.Vector3dVector(original_centered_pc))

    # output_pcd: new_num_points, 3
    # cubic_id_to_original_indices: new_num_points, max_indices_associated_with_single_cubic_pt
    start = time.time()
    output_pcd, cubic_id_to_original_indices, _ = pcd.voxel_down_sample_and_trace(voxel_size=0.004,
                                                      min_bound=obb.get_min_bound(),
                                                      max_bound=obb.get_max_bound())
    end = time.time()

    print("Time taken")
    print(end-start)

    new_dic = dic.copy()


    """
    Start Unit test
    """

    aggregate_uv1incam_depth_and_cam_idxs = create_aggregate_uv1incam_depth_and_cam_idxs\
                                                (uv_one_in_cam,
                                                 depths,
                                                 True,
                                                output_pcd=output_pcd,
                                                cubic_id_to_original_indices=cubic_id_to_original_indices
                                                )
    # partial_pcs = camera_util.convert_uv_depth_matrix_to_pointcloud(aggregate_uv1incam_depth_and_cam_idxs.copy(),
    #                                                                 cameras)

    # pc = np.concatenate(partial_pcs, axis=0)

    # pcd = vis_util.make_point_cloud_o3d(R.from_quat(dic['rotated_quat']).inv().apply(pc - dic['position']), [1., 0., 0.])
    # # visualize
    # o3d.visualization.draw_geometries([pcd,
    #                                    o3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0])])

    # original_centered_pc z and pc z should be around 1
    # assert np.all(np.isclose(pc - dic['position'], original_centered_pc))
    """
    End Unit test
    """

    assert aggregate_uv1incam_depth_and_cam_idxs.shape[0] < 10000
    new_dic['aggregate_uv1incam_depth_and_cam_idxs'] = aggregate_uv1incam_depth_and_cam_idxs.copy()

    """
    Unit test
    """
    print("ln150")

    partial_pcs = camera_util.convert_uv_depth_matrix_to_pointcloud(aggregate_uv1incam_depth_and_cam_idxs.copy(),
                                                                    cameras)

    pc = np.concatenate(partial_pcs, axis=0)

    # pc = pc-dic['position']
    # print("ln154")
    # pcd = vis_util.make_point_cloud_o3d(pc, [1., 0., 0.])
    # # visualize
    # open3d.visualization.draw_geometries([pcd,
    #                                       open3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0])])
    """
    End Unit Test
    """

    original_pc_in_canonical = R.from_quat(dic['rotated_quat']).inv().apply(original_centered_pc)

    new_dic['canonical_min_height'] = np.min(original_pc_in_canonical[:, -1])
    new_dic['canonical_max_height'] = np.max(original_pc_in_canonical[:, -1])
    # new_dic['rotated_pointcloud'] = fps_pc.copy()
    #
    del new_dic['original_rotated_centered_pointcloud']
    del new_dic['uv_one_in_cam']
    del new_dic['depths']
    torch.save(new_dic, new_samples_dir / row['object_name'] / f"{row['sample_idx']}.pkl")


# @synchronized
def generate_samples_from_canonical_pointclouds():
    """
    Takes in full pointcloud and processes it

    :return:
    """

    holder = dict()
    # for tupl in itertools.product(pcdset_original.data_df.iterrows(), range(num_fps_samples)):
    for tupl in itertools.product(pcdset_original.data_df.iterrows()):
        row_info = tupl[0]

        index = row_info[0]

        row = row_info[1]

        # fps_idx = tupl[1]

        dic = torch.load(row['file_path'])

        original_rotated_centered_pc = dic['original_rotated_centered_pointcloud']
        depths = dic['depths']
        uv_one_in_cam = dic['uv_one_in_cam']

        start = time.time()

        holder[str(index) + str(row)] = uvd_to_segmented_uvd(depths, uv_one_in_cam, row, dic, original_rotated_centered_pc)

        print("runtime uvd_to_sample_on_disk")
        print(time.time() - start)


generate_samples_from_canonical_pointclouds()