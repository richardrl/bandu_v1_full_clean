import numpy
import torch
# from bandu.utils import pb_util, vis_util, bandu_util
# from utils import surface_util, pointcloud_util, visualization_util
# from spinup.policies.handcrafted import handcrafted_policy1
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d
from utils.quaternion import qrot
from deco import *
from utils.vis_util import make_colors
from utils.surface_util import gen_surface_box
import pybullet as p
import time
from PIL import Image
from pathlib import Path
import json
import copy
# from bandu.imports.bingham_rotation_learning.qcqp_layers import A_vec_to_quat
import os
from utils import color_util, camera_util
# from imports.ipdf import models as ipdf_models


def evaluate_using_env(env, models_dict, model_device, pb_loop=False, max_episodes=100, display_loss=True,
                       stats_dic=None, urdf_ids_per_episode=None, ret_scores=False, vis_o3d=True, save_o3d=True,
                       use_gt_pts=False, use_pca_obb=False, img_render_dir=None, fps_num_points=2048, block_base=True,
                       linear_search=True, use_ransac_full_pc=False, gen_antiparallel_rotation=False):

    """

    :param env:
    :param models_dict:
    :param model_device:
    :param pb_loop:
    :param max_episodes: Max episodes to evaluate_using_env
    :param urdf_ids_per_episode: nB x (num_objects + 1) matrix of urdf object indicesindices
    :return:
    """
    cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                        camera_forward_z_offset=.2)
    if img_render_dir is not None:
        ird_path = Path(img_render_dir)
        ird_path.mkdir(exist_ok=True, parents=True)

    # assert env.scale_factor == pcs.args['global_scaling']
    stacked_scores = []
    for name, model in models_dict.items():
        model.eval()

    # terminal_states = np.where(serialize_dict['terminal'] == 1)[0]
    # starting_states_idxs = terminal_states + np.ones_like(terminal_states)

    # for idx in starting_states_idxs[:max_episodes]:
    # load starting setup for one episode
    # s = env.reset(sampled_path_idxs=serialize_dict['s']['sampled_path_idxs'][idx],
    #               serialized_ori=serialize_dict['s']['current_quats'][idx],
    #               serialized_pos=serialize_dict['s']['current_pos'][idx])

    if urdf_ids_per_episode is not None:
        max_episodes = len(urdf_ids_per_episode)

    for ep_idx in range(max_episodes):
        # env.current_urdf_tuple_idx = 19

        s = env.reset(sampled_path_idxs=urdf_ids_per_episode[ep_idx] if urdf_ids_per_episode is not None else None)
        print("\n\n RESET")
        if pb_loop:
            pb_util.pb_key_loop("n")
        try:
            if img_render_dir is not None:
                # ep_dir = Path(img_render_dir) / "_".join(s['object_names']) / str(ep_idx)
                ep_dir = Path(img_render_dir) / "_".join(s['object_names'])
                ep_dir.mkdir(exist_ok=True, parents=True)
                img = env.render(scale_factor=1/2)
                pil_img = Image.fromarray(img, 'RGB')

                pth = ep_dir / f"ep{ep_idx}_0.png"
                pil_img.save(pth)

            for timestep in range(env.num_objects + 2 + int(block_base)):
                # now run policy and do physics simulation
                # try:
                batch = get_batch_from_state(s, model_device, fps_num_points, stats_dic=stats_dic,
                                             linear_search=linear_search, cameras=cameras)
                # except Exception as e:
                #     print("ln83 e")
                #     print(e)
                #     break
                # if display_loss:
                #     gt_a = handcrafted_policy1(s)
                #     print(f"ln114 gt_a range(-1, num_objects): {int(gt_a['selected_object_id'])}")
                #     print(f"ln152 selected_object_id range(-1, num_objects): {int(selected_object_id.data.cpu())}")
                #
                #     target = gt_a['selected_object_id'] + 1
                #
                #     pred = predicted_logits.squeeze(dim=-1)
                #
                #     assert len(pred.shape) == 2, pred.shape
                #
                #     loss = F.cross_entropy(pred, torch.Tensor([target]).to(pred.device).long())
                #     print(f"ln122 single sample_idx loss: {float(loss.data.cpu())}")

                print(f"ln112 visited (before step): {s['visited']}")

                action_dict = dict()

                if "activevsnoop_classifier" in models_dict.keys():
                    selected_object_id = get_selected_oid(models_dict, batch)
                    action_dict['selected_object_id']= selected_object_id
                else:
                    selected_object_id = handcrafted_policy1(s, block_base=block_base)['selected_object_id']
                    action_dict['selected_object_id'] = selected_object_id

                if selected_object_id < 0:
                    assert timestep > 0
                    break

                sample_idx = 0
                canonical_pc = qrot(R.from_quat(batch['current_quats'][selected_object_id]).inv().as_quat(),
                                    batch['rotated_pointcloud'][sample_idx][selected_object_id])
                if use_pca_obb:
                    assert not use_gt_pts
                    if gen_antiparallel_rotation:
                        relative_rotmat, relative_rotmat2 = surface_util.get_relative_rotation_from_obb(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                                      vis_o3d=vis_o3d, gen_antiparallel_rotation=True)
                    else:
                        relative_rotmat = surface_util.get_relative_rotation_from_obb(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                              vis_o3d=vis_o3d)
                elif use_ransac_full_pc:
                    starting_pc_original = batch['rotated_pointcloud'][sample_idx][selected_object_id].detach()
                    starting_pc = batch['rotated_pointcloud'][sample_idx][selected_object_id]

                    starting_pc_device = copy.deepcopy(starting_pc.device)
                    # nB, nO, num_points, _ = starting_pc.shape
                    num_points = starting_pc.shape[0]

                    rr_arr, plane_model_arr, heights_arr = [], [], []

                    relative_rotmat_inner, plane_model_inner, plane_idxs = surface_util.get_relative_rotation_from_hot_labels(starting_pc,
                                                                                                                  torch.zeros(num_points),
                                                                                                                  ret_idxs=True)

                    rr_arr.append(relative_rotmat_inner)
                    plane_model_arr.append(plane_model_inner)

                    estimated_canonical_pc = R.from_matrix(relative_rotmat_inner).apply(starting_pc_original.data.cpu().numpy())

                    height_found = np.max(estimated_canonical_pc[:, -1])
                    print(height_found)

                    heights_arr.append(height_found)

                    while len(plane_idxs) > 50:
                        starting_pc = torch.as_tensor(np.delete(starting_pc.data.cpu().numpy(), plane_idxs, axis=0), device=starting_pc_device)

                        num_points = starting_pc.shape[0]
                        if num_points < 20:
                            break

                        # nB, nO, num_points, _ = starting_pc.shape
                        relative_rotmat_inner, plane_model_inner, plane_idxs = surface_util.get_relative_rotation_from_hot_labels(starting_pc,
                                                                                                                      torch.zeros(num_points),
                                                                                                                      ret_idxs=True)

                        rr_arr.append(relative_rotmat_inner)
                        plane_model_arr.append(plane_model_inner)

                        estimated_canonical_pc = R.from_matrix(relative_rotmat_inner).apply(starting_pc_original.data.cpu().numpy())

                        height_found = np.max(estimated_canonical_pc[:, -1])
                        print(height_found)
                        heights_arr.append(height_found)

                        canonical_pc = qrot(R.from_quat(batch['current_quats'][selected_object_id]).inv().as_quat(),
                                            starting_pc)
                        # geoms_to_draw = [
                        #     vis_util.make_point_cloud_o3d(starting_pc,
                        #                                   color=np.zeros(3))]
                        #     # vis_util.make_point_cloud_o3d(canonical_pc + np.array([0.,0.,.5]),
                        #     #                               color=np.zeros(3)),
                        #     # open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, -.5])]
                        #
                        # made_box, box_centroid = gen_surface_box(plane_model_inner, ret_centroid=True)
                        # geoms_to_draw.append(bandu_util.create_arrow(plane_model_inner[:3], [1., 0., 0.], position=box_centroid))
                        # geoms_to_draw.append(made_box)
                        #
                        # open3d.visualization.draw_geometries(geoms_to_draw)

                    best_id = np.argmax(heights_arr)
                    relative_rotmat = rr_arr[best_id]
                    plane_model = plane_model_arr[best_id]
                elif use_gt_pts:
                    # use the ground truth thresholded surface points
                    # relative_rotmat, plane_model = surface_util.get_relative_rotation_from_hot_labels(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                    #                                                                                   batch['bottom_thresholded_boolean'][selected_object_id].squeeze(-1))
                    relative_rotmat, plane_model = surface_util.get_relative_rotation_from_hot_labels(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                                                      batch['bottom_thresholded_boolean'][0, selected_object_id].squeeze(-1))
                elif "surface_classifier" in models_dict.keys() and selected_object_id > -1:

                    nB, nO, num_points, _ =  batch['rotated_pointcloud'].shape
                    # nB == 1, nO, 1024, 3 -> nB * nO, num_points -> nB, nO, num_points

                    print("ln216 memory stats")
                    print(torch.cuda.memory_summary(device=0, abbreviated=False))
                    print(torch.cuda.memory_summary(device=1, abbreviated=False))
                    if models_dict['surface_classifier'].label_type == "btb":
                        if "cvae" in models_dict['surface_classifier'].__class__.__name__.lower():
                            predicted_surface_binary_logits = models_dict['surface_classifier'].decode_batch(batch).reshape(nB, nO, num_points)
                        else:
                            predicted_surface_binary_logits = models_dict['surface_classifier'](batch).reshape(nB, nO, num_points)

                        relative_rotmat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                                          predicted_surface_binary_logits[sample_idx][selected_object_id])
                    else:
                        predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False, z_samples_per_sample=1)
                        if models_dict['surface_classifier'].A_vec_to_quat_head:
                            q = A_vec_to_quat(predictions[sample_idx])
                            relative_rotmat = R.from_quat(q.data.cpu().numpy()).as_matrix()[0]
                        else:
                            relative_rotmat = R.from_quat(predictions[sample_idx].data.cpu().numpy()).as_matrix()[0]

                    print("ln234 memory stats")
                    print(torch.cuda.memory_summary(device=0, abbreviated=False))
                    print(torch.cuda.memory_summary(device=1, abbreviated=False))
                    # batch['rotated_pointcloud'] shape: 1, 1, num_points, 3
                    # -> num_points', 3

                    # selected_surface_points = batch['rotated_pointcloud'][0][0][(torch.sigmoid(surface_binary_logits) < .5).squeeze(0)]
                    # surface_pcd = open3d.geometry.PointCloud()
                    # surface_pcd.points = open3d.utility.Vector3dVector(selected_surface_points.cpu().data.numpy())

                    # open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                    #                                                                     color=make_colors(surface_binary_logits.squeeze(0))),
                    #                                       vis_util.make_point_cloud_o3d(qrot(relquat, batch['rotated_pointcloud'][0][0]),
                    #                                                                     color=np.array([0.,1.,0.])),
                    #                                       vis_util.make_point_cloud_o3d(qrot(R.from_quat(batch['current_quats'][0]).inv().as_quat(), batch['rotated_pointcloud'][0][0]),
                    #                                                                     color=np.array([0.,0.,1.])),
                    #                                       gen_surface_box(plane_model),
                    #                                       open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])

                elif "ipdf" in models_dict.keys() and selected_object_id > -1:
                    # Apply models.
                    with torch.no_grad():
                        feature = models_dict['model'](batch['rotated_pointcloud'][sample_idx][selected_object_id].unsqueeze(0).permute(0, 2, 1))
                        # Recursion level 4 amounts to ~300k samples.
                        queries = ipdf_models.generate_queries(
                            num_queries=4,
                            mode='grid',
                            rotate_to=torch.eye(3, device=model_device)[None])
                        pdf, pmf = models_dict['ipdf'].compute_pdf(feature, queries)

                        # If we have to output a single rotation, this is it.
                        # TODO: we could run gradient ascent here to improve accuracy.
                        relative_rotmat = queries[0][pdf.argmax(axis=-1)][0].data.cpu().numpy()
                else:
                    raise NotImplementedError

                if vis_o3d or save_o3d:
                    if use_gt_pts or use_pca_obb:
                        # colors for the ground truth
                        # mc1 = make_colors(batch['bottom_thresholded_boolean'][selected_object_id].squeeze(-1),
                        #                  surface_color=[0., 1., 0.],
                        #                  background_color=[128 / 255, 0, 128 / 255])

                        # mc2 = make_colors(batch['bottom_thresholded_boolean'][selected_object_id].squeeze(-1))

                        mc1 = make_colors(batch['bottom_thresholded_boolean'][sample_idx, selected_object_id].squeeze(-1),
                                          background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)

                        mc2 = make_colors(batch['bottom_thresholded_boolean'][sample_idx, selected_object_id].squeeze(-1))
                    else:
                        if 'surface_classifier' in models_dict.keys() and models_dict['surface_classifier'].label_type == "btb":
                            # colors for the predicted contact points
                            # mc1 = make_colors(torch.sigmoid(surface_binary_logits[sample_idx][selected_object_id].squeeze(-1)),
                            #                   surface_color=[0., 1., 0.],
                            #                   background_color=[128 / 255, 0, 128 / 255])
                            mc2 = visualization_util.make_color_map(torch.sigmoid(predicted_surface_binary_logits[sample_idx][selected_object_id].squeeze(-1)))
                            mc1 = make_colors(torch.sigmoid(predicted_surface_binary_logits[sample_idx][selected_object_id].squeeze(-1)))

                    if 'surface_classifier' in models_dict.keys() and models_dict['surface_classifier'].label_type == "btb":
                        if not use_pca_obb:
                            geoms_to_draw = [
                                vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                              color=mc1),
                                # vis_util.make_point_cloud_o3d(canonical_pc + np.array([0.,0.,.5]),
                                #                               color=mc2),
                                open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])]

                            box, box_centroid = gen_surface_box(plane_model, ret_centroid=True, color=[0., 0., .5])
                            geoms_to_draw.append(bandu_util.create_arrow(plane_model[:3], [0., 0., .5],
                                                                         position=box_centroid,
                                                                         object_com=np.zeros(3)),
                                                 )
                            geoms_to_draw.append(box)

                            # geoms_to_draw.append(vis_util.make_point_cloud_o3d(s['table_pointcloud'], color=[0., 0., 1.]))
                            if vis_o3d:
                                open3d.visualization.draw_geometries(geoms_to_draw)

                                # temporary stuff for the presentation
                                open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                                                    color=mc1),
                                                                      open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                                open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                                                      color=[0., 0., 0.]),
                                                                      open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                        # open3d.visualization.draw_geometries([
                        #     vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][selected_object_id],
                        #                                   color=[0., 0., 0.]),
                        #     vis_util.make_point_cloud_o3d(canonical_pc + np.array([0.,0.,.5]),
                        #                                   color=[0., 0., 0.]),
                        #     open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, -.5])])

                def get_best_normal_best_theta(rel_quat, current_quat):
                    # target_quat
                    # target_rot = (R.from_quat(rel_quat) * R.from_quat(batch['current_quats'][selected_object_id]))
                    target_rot = (R.from_quat(rel_quat) * R.from_quat(current_quat))

                    # load URDF extra config
                    print("ln303 selected oid")
                    print(selected_object_id)
                    print(env.urdf_paths)

                    urdf_path = env.urdf_paths[env.sampled_path_idxs[selected_object_id]]
                    dir_ = os.path.dirname(urdf_path)
                    config_path = Path(dir_) / "extra_config"
                    with open(str(config_path), "r") as fp:
                        jd = json.load(fp)

                    # rotate normals according to target rot
                    rotated_normals = []

                    for normal_list in jd['normals']:
                        normal = np.array(normal_list)
                        rotated_normals.append(target_rot.apply(normal))

                    # find the normal that is closest to the gravity vector
                    thetas = []
                    for i in range(len(rotated_normals)):
                        theta = np.arccos(np.dot(rotated_normals[i], [0, 0, -1]))
                        thetas.append(theta)

                    min_idx = np.argmin(thetas)

                    # save that normal
                    best_normal = rotated_normals[min_idx]

                    # save the distance from that normal to the gravity rot
                    best_theta = thetas[min_idx]
                    return best_normal, best_theta

                if use_pca_obb and gen_antiparallel_rotation:
                    print("ln82 rrm")
                    print(relative_rotmat)
                    relquat = R.from_matrix(relative_rotmat).as_quat()

                    best_normal, best_theta = get_best_normal_best_theta(relquat, batch['current_quats'][selected_object_id])

                    relquat2 = R.from_matrix(relative_rotmat2).as_quat()

                    best_normal2, best_theta2 = get_best_normal_best_theta(relquat2, batch['current_quats'][selected_object_id])

                    if best_theta < best_theta2:
                        action_dict['relative_quat'] = relquat
                        # keep normals and quats
                    else:
                        # best_theta2 < best_theta
                        best_normal = best_normal2
                        best_theta = best_theta2
                        action_dict['relative_quat'] = relquat2
                else:
                    print("ln82 rrm")
                    print(relative_rotmat)
                    relquat = R.from_matrix(relative_rotmat).as_quat()
                    best_normal, best_theta = get_best_normal_best_theta(relquat, batch['current_quats'][selected_object_id])
                    action_dict['relative_quat'] = relquat

                print("ln325 best normal")
                print(best_normal)
                print(best_theta)

                s_new, r, d, info = env.step(action_dict, falling_reset=False, debug_rotation_angle=False,
                                             debug_draw_abb=False)

                print("\n\n\n")
                if save_o3d:
                    canonical_pc = qrot(R.from_quat(batch['current_quats'][selected_object_id]).inv().as_quat(),
                                        batch['rotated_pointcloud'][sample_idx][selected_object_id])
                    dic_it = dict(rotated_pointcloud=batch['rotated_pointcloud'][sample_idx][selected_object_id],
                                  canonical_pc=canonical_pc,
                                  btb=batch['bottom_thresholded_boolean'][sample_idx],
                                  batch=copy.deepcopy(batch),
                                  relative_rotmat=relative_rotmat,
                                  best_normal=best_normal,
                                  best_theta=best_theta
                                  )
                    if not use_pca_obb and 'surface_classifier' in models_dict.keys() and models_dict['surface_classifier'].label_type == "btb":
                        dic_it['mc1'] = mc1
                        dic_it['mc2'] = mc2
                        dic_it['plane_model']= plane_model
                        dic_it['box_centroid']=box_centroid,
                        dic_it['surface_binary_logits']=predicted_surface_binary_logits[sample_idx][selected_object_id].squeeze(-1)
                    torch.save(dic_it, ep_dir / f"ep{ep_idx}_{timestep}_o3d.pkl")

                # save one more time
                if img_render_dir is not None:
                    img = env.render(scale_factor=1/2)
                    pil_img = Image.fromarray(img, 'RGB')

                    pth = ep_dir / f"ep{ep_idx}_{s_new['timestep'][0]}.png"
                    pil_img.save(pth)

                s = s_new

                # if pb_loop:
                #     pb_util.pb_key_loop("n")

                if img_render_dir is not None and "info" in vars():
                    # ep_dir = Path(img_render_dir) / "_".join(s['object_names']) / str(ep_idx)
                    ep_dir = Path(img_render_dir) / "_".join(s['object_names'])

                    pth = ep_dir / f"ep{ep_idx}_{timestep}_info.json"
                    with open(pth, "w") as fp:
                        new_info = dict()

                        for k,v in info.items():
                            if isinstance(v, np.ndarray):
                                new_info[k] = v.tolist()
                            else:
                                new_info[k] = v

                        new_info['best_normal'] = best_normal.tolist()
                        new_info['best_theta'] = best_theta
                        json.dump(new_info, fp)

                print("\n\n")
                print(f"Final num stacked: {float(info['num_stacked'])}")
                stacked_scores.append(float(info['num_stacked']))
        except:
            # keep episodes coming
            continue

    for name, model in models_dict.items():
        model.train()

    if ret_scores:
        return np.mean(stacked_scores), stacked_scores
    else:
        return np.mean(stacked_scores)


def get_bti(batched_pointcloud,
            threshold_bottom_of_upper_region,
            threshold_top_of_bottom_region=None,
            max_z=None, min_z=None):
    """

    :param batched_pointcloud:
    :param threshold_bottom_of_upper_region: Bottom of upper region, expressed as fraction of total object height
    :param threshold_top_of_bottom_region: Top of bottom region, expressed as fraction of total object height
    :param line_search: search along the canonical axis for the section with the most support
    :return:
    """

    if threshold_top_of_bottom_region is not None:
        assert threshold_top_of_bottom_region < threshold_bottom_of_upper_region

    # returns boolean where 1s rotated_pc_mean BACKGROUND and 0 means surface

    # since pointcloud is in the canonical position, we chop off the bottom points
    if max_z is None and min_z is None:
        max_z = np.max(batched_pointcloud[..., -1], axis=-1)
        min_z = np.min(batched_pointcloud[..., -1], axis=-1)
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
    #     import pdb
    #     pdb.set_trace()
    #
    # else:
    # batch_size x num_objects x num_points
    # returns true for all points that are ABOVE the z-threshold
    bti = np.greater(batched_pointcloud[..., -1], np.expand_dims(threshold_world_bottom_of_upper_region, axis=-1))

    if threshold_top_of_bottom_region:
        threshold_world_top_of_bottom_region = min_z + object_heights * threshold_top_of_bottom_region
        bti_bottom_region = np.less(batched_pointcloud[..., -1], np.expand_dims(threshold_world_top_of_bottom_region, axis=-1))

        bti = bti + bti_bottom_region
    bti = bti[..., None]

    # assert np.sum(bti) > 0 and np.sum(bti) < batched_pointcloud.shape[0], np.sum(bti)
    return bti

@concurrent
def get_bti_from_rotated_concurrent(rotated_batched_pointcloud, orientation_quat, threshold_frac,
                                    max_z=None, min_z=None
                                    ):
    """
    Gets a single bti
    :param rotated_batched_pointcloud:
    :param orientation_quat:
    :param threshold_frac:
    :return:
    """
    assert len(rotated_batched_pointcloud.shape) == 2
    assert len(orientation_quat.shape) == 1

    # canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud.cpu().data.numpy())
    canonical = R.from_quat(orientation_quat).inv().apply(rotated_batched_pointcloud)

    return get_bti(canonical, threshold_frac, max_z=max_z, min_z=min_z)


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

        def find_bti(threshold_frac_inner):
            found_btis = []
            found_rotmats_distance_to_identity = []

            for lower_start_frac in np.linspace(0, max_frac_threshold, num=100):
                found_bti = get_bti(canonical, threshold_frac_inner + lower_start_frac, lower_start_frac, max_z=max_z, min_z=min_z)

                try:
                    relative_rotmat, plane_model = surface_util.get_relative_rotation_from_hot_labels(torch.as_tensor(canonical),
                                                                                                      torch.as_tensor(found_bti.squeeze(-1)),
                                                                                                      min_z=min_z,
                                                                                                      max_z=max_z)
                except:
                    continue
                # print(relative_rotmat)
                found_rotmats_distance_to_identity.append(np.linalg.norm(relative_rotmat - np.eye(3)))
                found_btis.append(found_bti)
            return found_rotmats_distance_to_identity, found_btis

        try:
            for threshold_frac_inner in [threshold_frac, 2*threshold_frac, 4*threshold_frac, 5*threshold_frac]:
                outer_found_rotmats_distance_to_identity, outer_found_btis = find_bti(threshold_frac_inner)

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


def get_batch_from_state(s, model_device, num_points, stats_dic=None, threshold_frac=1/50, linear_search=True,
                         cameras=None):
    """
    Gets batch from a single state from the environment
    :param s:
    :param model_device:
    :param num_Points: The number of FPS points is based on this object
    :param stats_dic:
    :return:
    """
    batch = dict()

    # urdf_paths = [env.urdf_paths[idx] for idx in s['sampled_path_idxs']]
    #
    # urdf_names = bandu_util.get_object_names(urdf_paths)

    # 1, nO, num_points, 3
    # batch['canonical_pointcloud'] = torch.Tensor(pcs_obj.get_canonicalized_pointclouds_ndarray(urdf_names)).unsqueeze(0).to(model_device)

    # 1, nO
    batch['visited'] = torch.Tensor(s['visited']).unsqueeze(0).to(model_device)

    # 1, nO
    batch['pybullet_object_ids'] = torch.Tensor(s['pybullet_object_ids']).unsqueeze(0).to(model_device)

    # batch['occluded_obj_id'] = torch.Tensor([urdf_name_to_id[name] for name in urdf_names]).unsqueeze(0).to(model_device)

    if stats_dic is not None:
        batch['rotated_pointcloud_mean'] = torch.as_tensor(stats_dic['rotated_pointcloud_mean']).to(model_device)
        batch['rotated_pointcloud_var'] = torch.as_tensor(stats_dic['rotated_pointcloud_var']).to(model_device)

    # nO, num_points, 3

    # pc = torch.Tensor(s['rotated_raw_pointcloud']).to(model_device)

    # center
    # s['current_pos']: 1, 3 -> 1, 1, 3
    # pc = pc - torch.Tensor(s['current_pos']).to(model_device).unsqueeze(0)

    # fps sampling
    # -> nO, num_points', 3

    centered_pc = []

    # if this line fails, then camera was off
    nO = len(s['rotated_raw_pointcloud'])
    print("ln348 num objects")
    print(nO)


    print("ln449 lengths")
    print(len(s['uv_one_in_cam']))
    print(len(s['depths']))

    for obj_id in range(nO):
        active_camera_ids = []
        for cam_id, dm in enumerate(s['depths'][obj_id]):
            if np.array(dm).size != 0:
                active_camera_ids.append(cam_id)

        print("ln449 num active")
        print(len(active_camera_ids))

        new_depths = [pointcloud_util.augment_depth_realsense(dm,
                                                              coefficient_scale=1)
                      for dm in [s['depths'][obj_id][cid] for cid in active_camera_ids]]

        new_uv_one_in_cams = [uvoc for uvoc in [s['uv_one_in_cam'][obj_id][_] for _ in active_camera_ids]]

        original_pc = camera_util.get_joint_pointcloud([cameras[id_] for id_ in active_camera_ids],
                                                       obj_id=None,
                                                       filter_table_height=False,
                                                       return_ims=False,
                                                       # rgb_ims=new_rgb_ims,
                                                       depth=new_depths,
                                                       uv_one_in_cam=new_uv_one_in_cams)

        # centered_pc.append(s['rotated_raw_pointcloud'][i] - s['current_pos'][i])
        centered_pc.append(original_pc - s['current_pos'][obj_id])

    fps_centered_pc = []
    np.random.seed(0)
    for pc in centered_pc:
        print("ln672 pc")
        print(pc[0])
        pc = pc[np.random.choice(pc.shape[0], 10000, replace=False, )]
        pc = pointcloud_util.get_farthest_point_sampled_pointcloud(pc, num_points)
        fps_centered_pc.append(pc)

        print("ln675 fps pc")
        print(pc[0])

    # torch.save(fps_centered_pc, "fps_centered_pc.pkl")
    # fps_centered_pc = torch.load("/home/richard/improbable/spinningup/fps_centered_pc.pkl")
    # add dummy batch index
    # -> nB == 1, nO, num_points', 3
    # pc = pc.unsqueeze(0)
    final_pc = torch.Tensor(fps_centered_pc).to(model_device).unsqueeze(0)

    # this is centered
    batch['rotated_pointcloud'] = final_pc

    # get all object PCs
    bhb = [get_bti_from_rotated(pc, s['current_quats'][obj_idx], threshold_frac,
                                linear_search=linear_search) for obj_idx, pc in enumerate(fps_centered_pc)]
    bhb = [torch.Tensor(bb).float().to(model_device) for bb in bhb]

    # -> nO, num_points, 1 -> 1, nO, num_points, 1
    bhb = torch.stack(bhb, dim=0).unsqueeze(0)

    batch['bottom_thresholded_boolean'] = bhb
    # batch['bottom_thresholded_boolean'] = torch.Tensor(get_bti_from_rotated_concurrent(pc[0][0], s['current_quats'][0],1/10)).float().to(model_device)
    # batch['bottom_thresholded_boolean'] = \
    #     torch.Tensor(get_bti_from_rotated_concurrent(pc, s['current_quats'],2/10)).float().to(model_device)

    batch['current_pos'] = s['current_pos']
    batch['current_quats'] = s['current_quats']
    return batch


def get_selected_oid(models_dict, batch):
    for name, model in models_dict.items():
        model.eval()
    with torch.no_grad():
        if len(models_dict.keys()) == 1:
            predicted_logits = next(iter(models_dict.items()))[1](batch)

            assert torch.argmax(predicted_logits.squeeze(dim=-1), dim=-1).shape == torch.Size([1])
            predicted_oid = torch.argmax(predicted_logits.squeeze(dim=-1), dim=-1)[0]
            return predicted_oid - 1
        else:
            # assume 2 models
            assert len(models_dict.keys()) == 2

            # -> nB, 2
            logits = models_dict['activevsnoop_classifier'](batch)
            assert len(logits.shape) == 2
            print(f"ln100 activevsnoop_classifier probabilities: {torch.softmax(logits, dim=-1).squeeze()}")

            # -> nB
            predicted_binary_classes = torch.argmax(logits, dim=-1)

            # collect all indices where the predicted class is 0 (noop)

            # TODO: technically, batchnorm is going to have wrong statistics...
            # so let's filter it?
            # no need because we use the running rotated_pc_mean
            # predicts from 0 to num_objects - 1
            predicted_oids_logits = models_dict['python_oid_predictor'](batch)
            print(f"ln111 python_oid_predictor probabilities: {torch.softmax(predicted_oids_logits, dim=1).squeeze()}")
            predicted_oids = torch.argmax(predicted_oids_logits.squeeze(-1), dim=-1)

            # the final output should be -1 if noop is selected
            # and predicted_oid[i] otherwise
            noop_indices = torch.where(predicted_binary_classes == 0)[0].long()
            predicted_oids[noop_indices] = -1
            return predicted_oids


def calculate_rotation_between_predicted_normal_and_ground_truth_set_of_normals(predicted_normal, object_name):
    # get the set of normals in the "stacked" pose based on object name
    # calculate the rotation between the predicted normal and each normal in the set
    # return the minimum

    # apply the pybullet rotation to our ground truth set of normal vectors
    # after we do that,
    pass