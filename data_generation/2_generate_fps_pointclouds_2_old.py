# load original dataset
# generates FPS and noisy pointclouds
from data_generation.dataset_old import PybulletPointcloudDataset
import torch
from utils.pointcloud_util import *
from utils import camera_util
from pathlib import Path
# from bandu.utils import *
import sys
from deco import *
import itertools
import time


# original_data_dir = "/home/richard/improbable/spinningup/out/canonical_pointclouds/bandu_val/test_v2/samples"
original_data_dir = sys.argv[1]
num_fps_samples = int(sys.argv[2])
randomize_noise = bool(int(sys.argv[3]))

pcdset_original = PybulletPointcloudDataset(original_data_dir)

new_samples_dir = Path(original_data_dir) / ".." / f"fps_randomizenoise{randomize_noise}_numfps{num_fps_samples}_samples"


object_names = pcdset_original.data_df.object_name.unique()

for obj_name in object_names:
    fd_dir = new_samples_dir / obj_name
    fd_dir.mkdir(parents=True, exist_ok=True)

cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                    camera_forward_z_offset=.2)


# @concurrent
def uvd_to_sample_on_disk(depths, uv_one_in_cam, row, fps_idx, dic, original_pc):
    """
    Add noise to UV-Depth, do FPS sampling, and save to pickle.

    :param depths: List[numpy.ndarray]
    :param uv_one_in_cam:
    :param row:
    :param fps_idx:
    :param dic:
    :param original_pc:
    :return:
    """
    if randomize_noise:
        active_camera_ids = []

        for cam_id, dm in enumerate(depths):
            if dm.size != 0:
                active_camera_ids.append(cam_id)

        new_depths = [augment_depth_realsense(dm, coefficient_scale=1)
                      for dm in [depths[cid] for cid in active_camera_ids]]

        out_dict = camera_util.get_joint_pointcloud([cameras[id_] for id_ in active_camera_ids],
                                                       obj_id=None,
                                                       filter_table_height=False,
                                                       return_ims=False,
                                                       # rgb_ims=new_rgb_ims,
                                                       depth=new_depths,
                                                       uv_one_in_cam=uv_one_in_cam)
        original_pc, depths, uv_one_in_cam = out_dict['aggregate_pointcloud'], out_dict['depth'], out_dict[
            'uv_one_in_cam']

        print(f"ln27 {row['sample_idx']} {fps_idx}")

    # center at COM
    original_pc = original_pc - dic['position']

    # uniform sample before FPS
    original_pc = original_pc[np.random.choice(original_pc.shape[0], 10000, replace=False)]

    fps_pc = get_farthest_point_sampled_pointcloud(original_pc,
                                                   2048)

    new_dic = dic.copy()

    original_pc_in_canonical = R.from_quat(dic['rotated_quat']).inv().apply(original_pc)

    new_dic['canonical_min_height'] = np.min(original_pc_in_canonical[:, -1])
    new_dic['canonical_max_height'] = np.max(original_pc_in_canonical[:, -1])
    new_dic['rotated_pointcloud'] = fps_pc.copy()

    del new_dic['original_rotated_centered_pointcloud']
    del new_dic['uv_one_in_cam']
    del new_dic['depths']
    torch.save(new_dic, new_samples_dir / row['object_name'] / f"{row['sample_idx'] + str(fps_idx)}.pkl")

# @synchronized
def generate_samples_from_canonical_pointclouds():
    """
    Takes in full pointcloud and processes it

    :return:
    """

    holder = dict()
    for tupl in itertools.product(pcdset_original.data_df.iterrows(), range(num_fps_samples)):
        row_info = tupl[0]

        index = row_info[0]

        row = row_info[1]

        fps_idx = tupl[1]

        dic = torch.load(row['file_path'])

        original_pc = dic['original_rotated_centered_pointcloud']
        depths = dic['depths']
        uv_one_in_cam = dic['uv_one_in_cam']

        start = time.time()
        holder[str(index) + str(row) + str(fps_idx)] = uvd_to_sample_on_disk(depths, uv_one_in_cam, row, fps_idx, dic,
                                                                             original_pc)

        print("runtime uvd_to_sample_on_disk")
        print(time.time() - start)

generate_samples_from_canonical_pointclouds()