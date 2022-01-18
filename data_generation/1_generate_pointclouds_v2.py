# You need to run this from the root of the project

import glob
from pathlib import Path
from bandu.config import TABLE_HEIGHT
import argparse
import os
import json
from deco import *
import open3d as o3d

import pybullet as p
import torch
import random
from scipy.spatial.transform import Rotation as R
from utils import camera_util, bandu_util, pointcloud_util, pointnet2_utils, pb_util

import hashlib
import time
import itertools
import numpy as np

@concurrent
def generate_and_save_canonical_sample(urdf_path, sample_idx, height_offset, global_scaling, pb_loop=False, simulate=True,
                                       compute_oriented_normals=False, o3d_viz=False, data_dir=None,
                                       object_name=None):
    """
    Generate canonical sample and save it to sample directory.

    :param urdf_path:
    :param sample_idx:
    :param height_offset:
    :param global_scaling:
    :param pb_loop:
    :param simulate:
    :param compute_oriented_normals:
    :param o3d_viz:
    :param data_dir:
    :param object_name:
    :return:
    """

    seed = int(hashlib.sha256(urdf_path.encode('utf-8')).hexdigest(), 16) % 10 ** 8 + sample_idx
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"ln21 urdf_path: {urdf_path}")

    # initial loading position
    position = [0., 0., TABLE_HEIGHT + .4]
    current_oid = p.loadURDF(urdf_path, basePosition=position,
                             baseOrientation=[0.,0.,0.,1.], globalScaling=global_scaling)

    initial_orientation = bandu_util.get_initial_orientation("full")

    # true dropping position
    start_pos = bandu_util.get_position_for_drop_above_table(height_off_set=height_offset)

    p.resetBasePositionAndOrientation(current_oid, start_pos, initial_orientation.as_quat())

    # current_oid = bandu_util.load_bandu_objects(urdf_paths=[urdf_path], scale_factor=.5)

    print("Num bodies")
    print(p.getNumBodies())

    if pb_loop:
        pb_util.pb_key_loop('n')


    if simulate:
        for _ in range(1000):
            p.stepSimulation()

    current_p, current_q = p.getBasePositionAndOrientation(current_oid)
    # pointcloud, depths, uv_one_in_cam = camera_util.get_joint_pointcloud(cameras,
    #                                               obj_id=current_oid,
    #                                               filter_table_height=False,
    #                                             return_uv_cam_only=True)

    out_dict = camera_util.get_joint_pointcloud(cameras,
                                                  obj_id=current_oid,
                                                  filter_table_height=False,
                                                return_uv_cam_only=True)
    pointcloud, depths, uv_one_in_cam = out_dict['aggregate_pointcloud'], out_dict['depth'], out_dict['uv_one_in_cam']

    # todo why did we have this
    # if not uv_one_in_cam:
    #     return dict()

    active_camera_ids = []

    # depths[0].shape: 3, 48885
    # uv_one_in_cam[0].shape: 48885
    # pointcloud.shape: 154259, 3
    for cam_id, dm in enumerate(depths):
        if dm.size != 0:
            active_camera_ids.append(cam_id)

    # apply realsense noise
    new_depths = [pointcloud_util.augment_depth_realsense(dm,
                                                             coefficient_scale=1)
                     for dm in [depths[cid] for cid in active_camera_ids]]
    # new_seg_ims = [seg_ims[cid] for cid in active_camera_ids]
    #
    # new_rgb_ims = [rgb_ims[cid] for cid in active_camera_ids]

    start = time.time()
    print("ln76 noise generation elapsed time")

    out_dict = camera_util.get_joint_pointcloud([cameras[id_] for id_ in active_camera_ids],
                                                                             obj_id=current_oid,
                                                                             filter_table_height=False,
                                                                             return_ims=False,
                                                                             # rgb_ims=new_rgb_ims,
                                                                            depth=new_depths,
                                                                            uv_one_in_cam=uv_one_in_cam
                                                                             # seg_ims=new_seg_ims
                                                                             )

    noisy_pc = out_dict['aggregate_pointcloud']

    end = time.time()
    print(end - start)

    if o3d_viz:
        npcd = vis_util.make_point_cloud_o3d(noisy_pc + np.array([0, 0., 0.5]), [0., 1., 0.])

        pcd = vis_util.make_point_cloud_o3d(pointcloud, [1., 0., 0.])

        o3d.visualization.draw_geometries([pcd, npcd])

    # compute vertex normals
    if compute_oriented_normals:
        pcd = vis_util.make_point_cloud_o3d(pointcloud, [1., 0., 0.])

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))

        pcd.normals = o3d.utility.Vector3dVector(mesh_util.fix_normal_orientation(np.array(pcd.points),
                                                       np.array(pcd.normals), np.array(current_p)))

        if o3d_viz:
            pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))
            o3d.visualization.draw_geometries([pcd])

    p.removeBody(current_oid)

    out_dic = dict(
        original_rotated_centered_pointcloud=pointcloud - current_p,
        position=np.array(current_p),
        rotated_quat=np.array(current_q),
        depths=depths,
        uv_one_in_cam=uv_one_in_cam
        # seg_ims=seg_ims
    )

    torch.save(out_dic, data_dir / object_name / f"{sample_idx}.pkl")
    return out_dic



@synchronized
def generate_urdf_name_to_pointcloud_dict(urdf_name_to_pointcloud_dict, urdf_dir, prefix, num_samples, urdfs, pointcloud_output_dir,
                                          height_offset=.2,
                                          global_scaling=1.5,
                                          simulate=True,
                                          compute_oriented_normals=True,
                                          pb_loop=False,
                                          o3d_viz=False):
    """
    Generates canonical samples for each urdf name. In the inner loop, saves the canonical samples.
    :param urdf_name_to_pointcloud_dict:
    :param urdf_dir:
    :param prefix:
    :param num_samples:
    :param urdfs:
    :param pointcloud_output_dir:
    :param height_offset:
    :param global_scaling:
    :param simulate:
    :param compute_oriented_normals:
    :param pb_loop:
    :param o3d_viz:
    :return:
    """

    assert urdfs

    canonical_samples_data_dir = Path(pointcloud_output_dir) / os.path.basename(urdf_dir) / prefix / "canonical_pointcloud_samples"

    print("ln181 saving to data_dir")
    print(canonical_samples_data_dir)

    # if we don't have the umask(0) and mode, the folder will be created with restrictive permissions
    os.umask(0)
    canonical_samples_data_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    # make folders for each object
    for obj_name in urdf_name_to_pointcloud_dict.keys():
        fd_dir = canonical_samples_data_dir / obj_name
        fd_dir.mkdir(parents=True, exist_ok=True)

    for (urdf_path, sample_idx) in itertools.product(urdfs, range(num_samples)):
        object_name = bandu_util.get_object_names([urdf_path])[0]
        urdf_name_to_pointcloud_dict[object_name][sample_idx] = generate_and_save_canonical_sample(urdf_path, sample_idx, height_offset,
                                                                                                   global_scaling, simulate=simulate,
                                                                                                   compute_oriented_normals=compute_oriented_normals,
                                                                                                   pb_loop=pb_loop, o3d_viz=o3d_viz,
                                                                                                   data_dir=canonical_samples_data_dir,
                                                                                                   object_name=object_name)

    print(urdf_name_to_pointcloud_dict)

    # cp_dir = Path(pointcloud_output_dir) / os.path.basename(urdf_dir) / prefix
    # cp_dir.mkdir(parents=True, exist_ok=True)
    #
    # urdf_name_to_pc_dict_path = str(cp_dir / f"urdf_name_to_pointcloud_dict")

    # save each tuple into the dict
    with open(str(Path(pointcloud_output_dir) / os.path.basename(urdf_dir) / prefix / f"args.json"), "w") as fp:
        settings_d = dict()
        settings_d['height_offset'] = height_offset
        settings_d['global_scaling'] = global_scaling
        settings_d['num_samples'] = num_samples
        settings_d['urdf_dir'] = urdf_dir
        json.dump(settings_d, fp, indent=4)

    # return urdf_name_to_pc_dict_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf_dir')
    parser.add_argument('prefix')
    parser.add_argument('--height_offset', type=float, default=.4, help="Height offset from the table when you drop it")
    parser.add_argument('--global_scaling', type=float, default=1.5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--pb_loop', action='store_true', help="Visualize pb and use keys to step forward")
    parser.add_argument('--compute_oriented_normals', action='store_true')
    parser.add_argument('--show_cams', action='store_true')
    parser.add_argument('--pc_save_dir', help="Save dir for canonical pointclouds samples",
                        default="out/canonical_pointclouds")
    parser.add_argument('--no_table', action='store_true')
    parser.add_argument('--no_simulate', action='store_true')
    parser.set_defaults(simulate=True)
    parser.set_defaults(table=True)

    args = parser.parse_args()

    plane_id, table_id = bandu_util.load_scene(p_connect_type=p.GUI if args.pb_loop else p.DIRECT, realtime=0)

    # p.removeBody(plane_id)
    if args.no_table:
        p.removeBody(table_id)

    cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                        camera_forward_z_offset=.2)

    if args.show_cams:
        for cam_id, cam in enumerate(cameras):
            cam_trans = cam.get_cam_ext()[:3,3]
            cam_rot = cam.get_cam_ext()[:3, :3]

            # visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
            #                                     rgbaColor=[1, 0, 0, .7],
            #                                         radius=0.1)
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName="/home/richard/d435.stl",
                                                meshScale=np.ones(3)*.002)
            collisionShapeId = -1  #p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="duck_vhacd.obj", collisionFramePosition=shift,meshScale=meshScale)

            # collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
            #                                           fileName="/home/richard/d435.stl",
            #                                           collisionFramePosition=np.zeros(3),
            #                                           meshScale=np.ones(3)*.002)

            print("ln195 p getNumBodies")

            print(p.getNumBodies())
            mb = p.createMultiBody(baseMass=1,
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=cam_trans,
                                   baseOrientation=R.from_matrix(cam_rot).as_quat()
                                   )

            print(p.getNumBodies())
            # p.addUserDebugText(f"{cam_id}", cam_trans + np.array([0., 0., 0.5]))
        pb_util.pb_key_loop("n")

    if not args.prefix:
        print("ln16 warning, prefix is not set, may overwrite existing dicts")

    urdfs = [f for f in glob.glob(str(Path(args.urdf_dir) / "**/*.urdf"), recursive=True)]

    assert urdfs, "Are you sure your urdf path is correct? If it's relative, check from what dir your script is running"

    urdf_name_to_pointcloud_dict = dict()
    for urdf_path in urdfs:
        object_name = bandu_util.get_object_names([urdf_path])[0]
        urdf_name_to_pointcloud_dict[object_name] = [None] * args.num_samples

    generate_urdf_name_to_pointcloud_dict(urdf_name_to_pointcloud_dict,
                                          args.urdf_dir,
                                          args.prefix,
                                          args.num_samples,
                                          urdfs,
                                          args.pc_save_dir,
                                          compute_oriented_normals=args.compute_oriented_normals,
                                          pb_loop=args.pb_loop,
                                          height_offset=args.height_offset,
                                          simulate=args.simulate)