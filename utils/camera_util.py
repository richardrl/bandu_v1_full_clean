import glob

import numpy as np
import pybullet as p
from matplotlib import pyplot as plt

from imports.airobot.sensor.camera import rgbdcam_pybullet
from utils.bullet_util import get_cam_img
from bandu.config import *

def setup_cameras(cam_pkls=BANDU_ROOT / "cameras/*.pkl",
                  dist_from_eye_to_focus_pt=1,
                  camera_forward_z_offset=0,
                  intrinsics_matrix=None):
    """
    
    :param cam_pkls: GLOB STRING, not a folder
    :param dist_from_eye_to_focus_pt: 
    :param camera_forward_z_offset: 
    :return: 
    """
    cam_pkls_paths = sorted([f for f in glob.glob(str(cam_pkls))], key=lambda str_x: str_x.split(".pkl")[0][-1])

    assert cam_pkls_paths

    # Cameras in the airobot object
    airobot_cameras = [rgbdcam_pybullet.RGBDCameraPybullet(None, p) for i in range(len(cam_pkls_paths))]

    # Setup the cameras
    [airobot_cameras[i].setup_camera_from_pkl(cam_pkls_paths[i],
                                              dist_from_eye_to_focus_pt=dist_from_eye_to_focus_pt,
                                              camera_forward_z_offset=camera_forward_z_offset,
                                              intrinsics_matrix=intrinsics_matrix) for i in range(len(airobot_cameras))]
    return airobot_cameras


def get_joint_pointcloud(airobot_cameras,
                         obj_id,
                         filter_table_height=True,
                         uniform_sample_max=None,
                         return_ims=False,
                         return_uv_cam_only=False,
                         # rgb_ims=None,
                         depth=None,
                         # seg_ims=None,
                         ignore_obj_id=None,
                         uv_one_in_cam=None,
                         augment_extrinsics=False,
                         object_com = None):
    """

    If this function fails due to incorrect outputs, check that you have the right return_ims or return_uv_cam_only falgs.

    :param airobot_cameras:
    :param obj_id:
    :param filter_table_height:
    :param uniform_sample_max:
    :param return_ims:
    :param rgb_ims: If the three ims are provided, then we can reconstruct
    :param depth:
    :param seg_ims:
    :return:
    """
    assert airobot_cameras, airobot_cameras

    """

    :param airobot_cameras:
    :return: Numpy array of (Num points X 3)
    """
    # List of tuples (pts, colors)
    top_of_table_height = TABLE_HEIGHT

    outputs = [cam.get_pcd(in_world=True,
                           filter_depth=filter_table_height,
                           obj_id=obj_id,
                           ignore_obj_id=ignore_obj_id,
                           depth_min=top_of_table_height-.01 if filter_table_height else None,
                           return_ims=return_ims,
                           uv_one_in_cam=uv_one_in_cam[cam_id] if uv_one_in_cam is not None else None,
                           # rgb_im=rgb_ims[cam_id] if rgb_ims is not None else None,
                           # seg_im=seg_ims[cam_id] if seg_ims is not None else None,
                           depth=depth[cam_id] if depth is not None else None,
                           return_uv_cam_only=return_uv_cam_only)
               for cam_id, cam in enumerate(airobot_cameras)]

    # Each point cloud is num_points X 3
    pointclouds = [output['pcd_pts'] for output in outputs]

    camera_idx_labels = [np.ones_like(output['pcd_pts'])*output_idx for output_idx, output in enumerate(outputs)]

    # if augment_extrinsics:
    #     # apply random transform about the object COM
    #     # should be small rotation, larger rotation
    #     pointclouds_tmp = []
    #
    #     for partial_pc in pointclouds:
    #         centered_partial_pc = partial_pc - object_com
    #
    #         # randomize translation
    #         centered_partial_pc += (np.random.uniform(3) - .5) * np.array([.04, .04, .005])
    #
    #         pointclouds_tmp.append(centered_partial_pc + object_com)
    #
    #     pointclouds = pointclouds_tmp

    return dict(
        aggregate_pointcloud=np.concatenate(pointclouds, axis=0),
        camera_idx_labels=camera_idx_labels,
        depth=[output['depth'] for output in outputs],
        uv_one_in_cam=[output['uv_one_in_cam'] for output in outputs]
    )
    # if return_uv_cam_only:
    #     depth = [output[-2] for output in outputs]
    #     uv_one_in_cam = [output[-1] for output in outputs]
    #
    #     return np.concatenate(pointclouds, axis=0), depth, uv_one_in_cam
    #
    # if return_ims:
    #     rgb_ims = [output[-3] for output in outputs]
    #     depth = [output[-2] for output in outputs]
    #     seg_ims = [output[-1] for output in outputs]
    #     return np.concatenate(pointclouds, axis=0), rgb_ims, depth, seg_ims
    #
    # if uniform_sample_max is not None:
    #     try:
    #         return pointcloud_util.uniform_downsample(uniform_sample_max, np.concatenate(pointclouds, axis=0))
    #     except:
    #         return np.concatenate(pointclouds, axis=0)
    # else:
    #     return np.concatenate(pointclouds, axis=0)


def save_image(
        image_name,
        cam_pkl=None,
        img_save_path=None,
        scale_factor=1/2,
):
    assert img_save_path is not None
    np_image = get_image(cam_pkl, scale_factor)
    plt.imsave(img_save_path + "/" + image_name, np_image)


def get_image(
        cam_pkl=None,
        scale_factor=1/2,
):
    if cam_pkl is None:
        cam_pkl = str(BANDU_ROOT / "pybullet/cams/cam0.pkl")

    img_width, img_height, rgbPixels = get_cam_img(cam_pkl, scale_factor)[:3]
    np_image = np.reshape(rgbPixels, (img_height,img_width, 4))[:, :, :3].astype(np.uint8)
    return np_image


def convert_depth_to_pointcloud(depth_ims, camera_ext, camera_int):
    """
    This only works if there was no image segmentation done

    :param depth_ims: m x n
    :param camera_ext: 4 x 4
    :param camera_int: 3 x 3
    :return:
        pointcloud: (mxn) x 3
    """
    img_height, img_width = depth_ims.shape
    uv_coords = np.mgrid[0: img_height,
               0: img_width].reshape(2, -1)
    uv_coords[[0, 1], :] = uv_coords[[1, 0], :]

    p_uv_one_C = np.concatenate((uv_coords,
                              np.ones((1, uv_coords.shape[1]))))

    p_uv_one_C = np.dot(np.linalg.inv(camera_int), p_uv_one_C)

    p_scene_C = np.multiply(p_uv_one_C, depth_ims.flatten())

    p_scene_one_C = np.concatenate((p_scene_C,
                    np.ones((1, p_scene_C.shape[1]))),
                   axis=0)

    p_scene_one_W = np.dot(camera_ext, p_scene_one_C)

    return p_scene_one_W[:3, :].T


def convert_uv_depth_to_pointcloud(p_uv_one_C, depth_ims_flattened, camera_ext):
    """

    Args:
        p_uv_one_C: [3, num_points]
        depth_ims_flattened: [num_points]
        camera_ext: [4, 4]

    Returns:

    """
    p_scene_C = np.multiply(p_uv_one_C, depth_ims_flattened)

    p_scene_one_C = np.concatenate((p_scene_C,
                    np.ones((1, p_scene_C.shape[1]))),
                   axis=0)

    p_scene_one_W = np.dot(camera_ext, p_scene_one_C)

    return p_scene_one_W[:3, :].T


def convert_uv_depth_matrix_to_pointcloud(aggregate_uv1incam_depth_and_cam_idxs, cameras):
    """

    :param aggregate_uv1incam_depth_and_cam_idxs: (2048, 5)
        First three dimensions are uv1
        Fourth dimension is depth
        Fifth dimension is camera idx
    :param cameras:
    :return:
    """
    partial_pcs = []
    for cam_idx, cam in enumerate(cameras):
        one_cam_data = aggregate_uv1incam_depth_and_cam_idxs[aggregate_uv1incam_depth_and_cam_idxs[:, -1] == cam_idx]
        partial_pc = convert_uv_depth_to_pointcloud(one_cam_data[:, :3].T, #uvone
                                       one_cam_data[:, 3].T, # depth
                                       cam.cam_ext_mat)

        # from utils import vis_util
        # import open3d as o3d
        # pcd = vis_util.make_point_cloud_o3d(partial_pc, [1., 0., 0.])
        # # visualize
        # o3d.visualization.draw_geometries([pcd])

        partial_pcs.append(partial_pc)
    return partial_pcs
    # return np.concatenate(partial_pcs, axis=0)


if __name__ == "__main__":
    # unit test for pickle loading
    import torch
    import sys
    import open3d
    import pickle

    canonical_pkl = torch.load(sys.argv[1])

    cam_pkls = ["out/0_cam.pkl", "out/1_cam.pkl", "out/2_cam.pkl", "out/3_cam.pkl"]

    cameras = []
    for cam_pkl in cam_pkls:
        with open(cam_pkl, "rb") as fp:
            cam = pickle.load(fp)
            cameras.append(cam)

    from utils.vis_util import make_point_cloud_o3d
    pc = convert_uv_depth_to_pointcloud(canonical_pkl['uv_one_in_cam'][0],
                                        canonical_pkl['depths'][0],
                                        cameras[0]['cam_ext_mat'])

    open3d.visualization.draw_geometries([make_point_cloud_o3d(pc, [0, 0, 0])])



    # unit test for converting from matrix
