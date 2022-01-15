# load two models
# use first model to predict the ID
# if first model predicts action, use second model to predict which object

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hyper_config', help='Manually combine the two previous hyper configs, for now')
parser.add_argument('env_config', type=str, help="String to path for env config file")

parser.add_argument('--anc_checkpoint', help="Model checkpoint: activevsnoop_classifier")
parser.add_argument('--pop_checkpoint', help="Model checkpoint: python_oid_predictor")
parser.add_argument('--sc_checkpoint', help="Model checkpoint: surface_classifier")
parser.add_argument('--pkl_dir', help="Used only to load the stats dic right now")

parser.add_argument('--serialization_pkl', help="Data checkpoint")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--pb_loop', action='store_true')
parser.add_argument('--vis_o3d', action='store_true')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--loss_str', default="ce", type=str, help="Loss str describing what loss fnx to use")
parser.add_argument('--max_partitions', type=int, default=20, help="Max number of batches to train each epoch")
parser.add_argument('--seed', type=int, default=0, help='Torch and Numpy random seed')
parser.add_argument('--num_points', type=int, default=150, help="Num points for FPS sampling")
parser.add_argument('--num_fps_samples', type=int, default=1, help="Num samples for FPS sampling")
parser.add_argument('--resume_initconfig', type=str, help="Initconfig to resume from")
parser.add_argument('--train_ep_frac', type=float, default=1.0, help="How many training episodes to use. "
                                                                     "Make sure this matches the argument during training.")
parser.add_argument('--check_samples', action='store_true', help='Check if the samples in play of the initconfig return '
                                                                 'those generated by the get samples in play function')
parser.add_argument('--no_model', action='store_true', help='Dont use a model. Save GPU')
parser.add_argument('--phase', type=str, default='test', help="Which phase to use in the env")
parser.add_argument('--stats_json', type=str)
parser.add_argument('--results_dir', default="out/env_eval/", type=str, help="Name of results directory")
parser.add_argument('--gpu0', type=int, default=0, help="GPU ID to use for multi GPU 0")
parser.add_argument('--gpu1', type=int, default=1, help="GPU ID to use for multi GPU 1")
parser.add_argument('--save_o3d', action='store_true')
parser.add_argument('--use_pca_obb', action='store_true')
parser.add_argument('--use_ransac_full_pc', action='store_true')
parser.add_argument('--use_gt_pts', action='store_true')
parser.add_argument('--block_base', action='store_true')
parser.add_argument('--gen_antiparallel_rotation', action='store_true')

args = parser.parse_args()

from envs.bandu_noop import BanduNoopEnv
from utils.misc_util import *
from utils.train_util import model_creator
from utils.env_evaluation_util import evaluate_using_env
import importlib
import json
from pathlib import Path
import pybullet as p

spec = importlib.util.spec_from_file_location("env_config", args.env_config)
env_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_config)
env_params = env_config.env_params

import torch
torch.random.manual_seed(args.seed)

import numpy
numpy.random.seed(args.seed)

import random
random.seed(args.seed)

if args.no_model:
    models_dict = dict()
    stats_dic = None
    MODEL_DEVICE = None
else:
    config=load_hyperconfig_from_filepath(args.hyper_config)

    config['model0']['model_kwargs']['gpu0'] = args.gpu0
    config['model0']['model_kwargs']['gpu1'] = args.gpu1

    models_dict = model_creator(config=config,
                                device_id=args.device_id)

    if args.anc_checkpoint:
        # load activevsnoop
        sd = torch.load(args.anc_checkpoint, map_location="cpu")
        models_dict['activevsnoop_classifier'].load_state_dict(sd['model'])

    if args.pop_checkpoint:
        # load active only oid predictor
        sd = torch.load(args.pop_checkpoint, map_location="cpu")
        models_dict['python_oid_predictor'].load_state_dict(sd['model'])

    if args.sc_checkpoint:
        sd = torch.load(args.sc_checkpoint, map_location="cpu")
        models_dict['surface_classifier'].load_state_dict(sd['model'])

        models_dict['surface_classifier'].gpu_0 = torch.device(f"cuda:{args.gpu0}")
        models_dict['surface_classifier'].gpu_1 = torch.device(f"cuda:{args.gpu1}")

    MODEL_DEVICE = next(next(iter(models_dict.items()))[1].parameters()).device
    if args.pkl_dir:
        # calculate aggregate loss over serialization set
        serialization_dict, dset_params = get_sd_and_params_from_pkl_dir(args.pkl_dir)

        env_params['num_objects'] = dset_params['num_objects']
        env_params['urdf_dir'] = dset_params['env_params']['urdf_dir']

        with open(args.stats_json, "r") as fp:
            stats_dic = json.load(fp)

        test_samples_in_play = get_samples_in_play(serialization_dict, dset_params, args.train_ep_frac, return_batched=True,
                                                   phase=args.phase)
        sample_idx_per_episode = [vec[0] for vec in test_samples_in_play]

        test_urdf_ids_per_episode = serialization_dict['s']['sampled_path_idxs'][sample_idx_per_episode].astype(int)
    else:

        with open(args.stats_json, "r") as fp:
            stats_dic = json.load(fp)

if args.pb_loop:
    env_params['p_connect_type'] = p.GUI


env = BanduNoopEnv(**env_params)


# if this line fails, check that you provided resume_initconfig
if args.resume_initconfig:
    pcs = torch.load(args.resume_initconfig)['pcs'] if args.resume_initconfig \
        else PointcloudSampler(args.num_points, args.num_fps_samples,
                               urdf_name_to_pc_path=dset_params['urdf_name_to_pointcloud_dict_path'])

# -> num_episodes x (num_objects + 1)
# checks if the samples in play are the same as the ones recorded in the initconfig during the previous training run
if args.check_samples:
    assert args.resume_initconfig
    train_samples = get_samples_in_play(serialization_dict, dset_params, args.train_ep_frac, return_batched=False,
                        phase="train")
    assert set(torch.load(args.resume_initconfig)['samples_in_play']) == set(train_samples), "Are you sure 1) " \
                                                                                             "your dataset is correct " \
                                                                                             "and 2) your ep_frac is correct"



rd_path = Path(args.results_dir)
rd_path.mkdir(exist_ok=True, parents=True)

with open(Path(args.results_dir) / "args.json", "w") as fp:
    json.dump(vars(args), fp, indent=4)

mean_s, scores = evaluate_using_env(env, models_dict, MODEL_DEVICE, pb_loop=args.pb_loop, max_episodes=22 * 100,
                                    stats_dic=stats_dic, urdf_ids_per_episode=test_urdf_ids_per_episode[:100]
    if "test_urdf_ids_per_episode" in locals() else None, ret_scores=True, vis_o3d=args.vis_o3d, save_o3d=args.save_o3d,
                                    use_gt_pts=args.use_gt_pts, use_pca_obb=args.use_pca_obb, img_render_dir=f"{args.results_dir}",
                                    fps_num_points=2048, block_base=args.block_base, use_ransac_full_pc=args.use_ransac_full_pc,
                                    gen_antiparallel_rotation=args.gen_antiparallel_rotation,
                                    augment_extrinsics=True)
print(f"Average stacked: {mean_s}")

with open(f"../out/out/phase{args.phase}_scores_meanstacked{mean_s}.json", "w") as fp:
    json.dump(scores, fp)