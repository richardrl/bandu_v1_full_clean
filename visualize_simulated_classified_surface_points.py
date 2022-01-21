import argparse

from utils.vis_util import make_color_map, make_colors
from utils import vis_util, color_util

parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('resume_pkl', type=str, help="Checkpoint to resume from")
parser.add_argument('train_dset_path', type=str)

parser.add_argument('--device_id', default=0)
parser.add_argument('--num_points', type=int, default=150, help="Num points for FPS sampling")
parser.add_argument('--num_fps_samples', type=int, default=1, help="Num samples for FPS sampling")
parser.add_argument('--resume_initconfig', type=str, help="Initconfig to resume from")
parser.add_argument('--stats_json', type=str)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--encoder_recon', action='store_true', help="Use encoder for reconstruction")

parser.add_argument('--val_dset_path', type=str)
parser.add_argument('--center_fps_pc', action='store_true', help='Center FPS')

args = parser.parse_args()


from utils.train_util import model_creator
# from supervised_training.utils.misc_util import *
# from supervised_training.utils.training_util import *
# from supervised_training.utils.loss_util import *
# from supervised_training.utils.pointcloud_util import PointcloudSampler
# from supervised_training.utils.batch_processors import process_batch_relativerotation
# from supervised_training.utils import surface_util, vae_util
# from bandu.utils import *
from scipy.spatial.transform.rotation import Rotation as R
from utils import misc_util, surface_util
import torch
import open3d
from torch.utils.data import DataLoader
from data_generation.dataset import PointcloudDataset
import json
import numpy as np

torch.set_printoptions(edgeitems=12)
config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)

# load model
models_dict = model_creator(config=config,
                            device_id=args.device_id)

device = torch.device("cuda:0")
sd = torch.load(args.resume_pkl, map_location=device)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)


models_dict['surface_classifier'].load_state_dict(sd['model'])

train_dset = PointcloudDataset(args.train_dset_path,
                               stats_dic=stats_dic,
                               center_fps_pc=args.center_fps_pc,
                               shear_aug=None,
                               scale_aug=None,
                               threshold_frac=.06,
                               max_frac_threshold=.2,
                               linear_search=True,
                                augment_extrinsics=True,
                               depth_noise_scale=1.5)
train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=args.batch_size, drop_last=True, shuffle=True)

batch = next(iter(train_dloader))

if "CVAE" in models_dict['surface_classifier'].__class__.__name__:
    if args.encoder_recon:
        predictions = models_dict['surface_classifier'](batch)['decoder'][0].unsqueeze(1)
        predictions_num_z = 1
    else:
        predictions_num_z = 1
        predictions = models_dict['surface_classifier'].decode_batch(batch, ret_eps=False,
                                                                     z_samples_per_sample=predictions_num_z)
else:
    predictions = models_dict['surface_classifier'](batch)

total_z_per_sample = 10

for sample_idx in range(args.batch_size):
    for z_idx in range(predictions_num_z):
        print("ln116 pred shape")
        print(predictions.shape)

        mat, plane_model = surface_util.get_relative_rotation_from_binary_logits(batch['rotated_pointcloud'][sample_idx][0],
                                                                                 predictions[sample_idx][z_idx])
        relrot = R.from_matrix(mat).as_quat()

        if "eps" in vars().keys():
            print("eps")
            print(eps[sample_idx])

        box, box_centroid = surface_util.gen_surface_box(plane_model, ret_centroid=True, color=[0, 0, .5])
        arrow = vis_util.create_arrow(plane_model[:3], [0., 0., .5],
                                                   position=box_centroid,
                                                   # object_com=sample_pkl['position'])
                                                   object_com=np.zeros(3))  # because the object has been centered

        open3d.visualization.draw_geometries([vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0],
                                                                            color=make_colors(torch.sigmoid(predictions[sample_idx][z_idx]))),
                                              vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, .5]),
                                                                            # color=make_colors(batch['bottom_thresholded_boolean'][sample_idx][0][:, 0],
                                                                            # background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                                                                            color=make_color_map(torch.sigmoid(predictions[sample_idx][z_idx].squeeze(-1)))),
                                              vis_util.make_point_cloud_o3d(batch['rotated_pointcloud'][sample_idx][0] + torch.tensor([0, 0, 1]),
                                                                            color=make_colors(batch['bottom_thresholded_boolean'][sample_idx],
                                                                                              background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)),
                                              box,
                                              arrow,
                                              open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])