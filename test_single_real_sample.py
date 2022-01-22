import torch
from utils import misc_util, train_util, surface_util, vis_util
import open3d
import argparse
import numpy as np
import json
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('sc_checkpoint', help="Model checkpoint: surface_classifier")
parser.add_argument('sample_path')
parser.add_argument('--stats_json')
parser.add_argument('--device_id', default=0)

args = parser.parse_args()


config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)

models_dict = train_util.model_creator(config=config,
                            device_id=args.device_id)

sd = torch.load(args.sc_checkpoint, map_location="cpu")

models_dict['surface_classifier'].load_state_dict(sd['model'])

# models_dict['surface_classifier'].gpu_0 = torch.device(f"cuda:{args.gpu0}")
# models_dict['surface_classifier'].gpu_1 = torch.device(f"cuda:{args.gpu1}")


# load sample
# sample_path = "/root/bandu_v1_full_clean/out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples/Knight Shape/1.pkl"

# sample_path = "/root/bandu_v1_full_clean/out/aggregate_pc.torch"

sample_pkl = torch.load(args.sample_path)

predictions_num_z = 1

batch = dict()

# num_points, 3 -> 1, 1, num_points, 3

# this is the real image pkl
pcd = vis_util.make_point_cloud_o3d(sample_pkl['points'],
                                    color=sample_pkl['colors'])

obb = open3d.geometry.OrientedBoundingBox()
obb = obb.create_from_points(pcd.points)

# center at COM
pcd.points = open3d.utility.Vector3dVector(np.array(sample_pkl['points']) - obb.get_center())

# visualize pointcloud
open3d.visualization.draw_geometries([pcd])

downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.004)
batch['rotated_pointcloud'] = torch.from_numpy(np.array(downsampled_pcd.points)).unsqueeze(0).unsqueeze(0)
# assert batch['rotated_pointcloud'].shape[2] > 1024 and batch['rotated_pointcloud'].shape[2] < 2048, batch['rotated_pointcloud'].shape

# center pointcloud
batch['rotated_pointcloud'] -= batch['rotated_pointcloud'].mean(axis=2)

# below is if we have the training file pkl
# batch['rotated_pointcloud'] = torch.from_numpy(sample_pkl['rotated_pointcloud']).unsqueeze(0).unsqueeze(0)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)
batch['rotated_pointcloud_mean'] = torch.Tensor(stats_dic['rotated_pointcloud_mean'])

batch['rotated_pointcloud_var'] = torch.as_tensor(stats_dic['rotated_pointcloud_var'])

models_dict['surface_classifier'].eval()

predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                             z_samples_per_sample=predictions_num_z)

# TODO: was this trained on threshold 0 or .5?
rotmat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][0][0],
                                                                            predictions[0][0])

geoms_to_draw = []


box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .5])
norm_arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                             position=box_centroid,
                                           # object_com=sample_pkl['position'])
                                             object_com=np.zeros(3)), # because the object has been centered
geoms_to_draw.append(norm_arrow
                     )
geoms_to_draw.append(box)

# original_rgb_with_red_contact_points =

surface_points_binary_mask = torch.round(torch.sigmoid(predictions[0][0]))

contact_points_binary_mask = 1 -surface_points_binary_mask

original_rgb_with_red_contact_points = (torch.from_numpy(np.array(downsampled_pcd.colors)).to(surface_points_binary_mask.device) * surface_points_binary_mask) + \
                                       contact_points_binary_mask * torch.Tensor([1, 0, 0]).unsqueeze(0).expand(surface_points_binary_mask.shape[0], -1).to(contact_points_binary_mask.device)

object_realsense_pcd = vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                                                    color=original_rgb_with_red_contact_points.data.cpu().numpy() )
geoms_to_draw.append(object_realsense_pcd)
                                                                    # color=vis_util.make_colors(
                                                                    #     torch.sigmoid(predictions[0][0])) ))
coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])
geoms_to_draw.append(coord_frame)


# icp visualization
mesh_path = "parts/stls/main/engmikedset/Nut.stl"

# resize to 0.75

object_mesh = open3d.io.read_triangle_mesh(mesh_path)

object_mesh.scale(.7, center=np.zeros(3))


object_mesh.paint_uniform_color(np.array([64,224,208])/255)
# object_mesh.paint_uniform_color(np.array([0, 0, 1]))

object_mesh.compute_vertex_normals()

object_mesh_pcd = object_mesh.sample_points_uniformly(number_of_points=1024)

geoms_to_draw.append(object_mesh)
mat = open3d.visualization.rendering.MaterialRecord()
mat.base_color = np.array([1, 1, 1, .8])
mat.shader = "defaultLitTransparency"


# open3d.visualization.draw_geometries(geoms_to_draw)

trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
threshold = 0.02

reg_p2p = o3d.pipelines.registration.registration_icp(
    object_mesh_pcd, object_realsense_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

open3d.visualization.draw([{'name': 'object_mesh', 'geometry': object_mesh.transform(reg_p2p.transformation), 'material': mat},
                           {'name': 'coordinate_frame', 'geometry': coord_frame},
                           {'name': 'object_realsense_pcd', 'geometry': object_realsense_pcd},
                           # {'name': 'norm_arrow', 'geometry': norm_arrow},
                           {'name': 'box', 'geometry': box}],
                          show_skybox=False)

# open3d.visualization.draw({'name': 'test', 'geometry': mesh, 'material': mat})