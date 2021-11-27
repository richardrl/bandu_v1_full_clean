import glob

import numpy as np
import pybullet as p
from matplotlib import pyplot as plt

from bandu.imports.airobot.sensor.camera import rgbdcam_pybullet
from utils.bullet_util import get_cam_img
from bandu.config import *

def setup_cameras(cam_pkls=BANDU_ROOT / "cameras/*.pkl",
                  dist_from_eye_to_focus_pt=1,
                  camera_forward_z_offset=0):
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
                                              camera_forward_z_offset=camera_forward_z_offset) for i in range(len(airobot_cameras))]
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
                         uv_one_in_cam=None):
    """

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
    pointclouds = [output[0] for output in outputs]

    if return_uv_cam_only:
        depth = [output[-2] for output in outputs]
        uv_one_in_cam = [output[-1] for output in outputs]
        return np.concatenate(pointclouds, axis=0), depth, uv_one_in_cam

    if return_ims:
        rgb_ims = [output[-3] for output in outputs]
        depth = [output[-2] for output in outputs]
        seg_ims = [output[-1] for output in outputs]
        return np.concatenate(pointclouds, axis=0), rgb_ims, depth, seg_ims

    if uniform_sample_max is not None:
        try:
            return pointcloud_util.uniform_downsample(uniform_sample_max, np.concatenate(pointclouds, axis=0))
        except:
            return np.concatenate(pointclouds, axis=0)
    else:
        return np.concatenate(pointclouds, axis=0)


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