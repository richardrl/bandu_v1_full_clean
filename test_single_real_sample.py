import torch
from utils import misc_util, train_util, surface_util, vis_util
import open3d
import argparse
import numpy as np
import json


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

batch['rotated_pointcloud'] = torch.from_numpy(np.array(pcd.voxel_down_sample(voxel_size=0.004).points)).unsqueeze(0).unsqueeze(0)
assert batch['rotated_pointcloud'].shape[2] > 1024 and batch['rotated_pointcloud'].shape[2] < 2048, batch['rotated_pointcloud'].shape

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
mat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][0][0],
                                                                                 predictions[0][0])

geoms_to_draw = []


box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .5])
geoms_to_draw.append(vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                             position=box_centroid,
                                           # object_com=sample_pkl['position'])
                                             object_com=np.zeros(3)), # because the object has been centered
                     )
geoms_to_draw.append(box)

geoms_to_draw.append(vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][0][0],
                                                                    color=vis_util.make_colors(
                                                                        torch.sigmoid(predictions[0][0]))))

geoms_to_draw.append(open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0]))
open3d.visualization.draw_geometries(geoms_to_draw)
# run model on sample