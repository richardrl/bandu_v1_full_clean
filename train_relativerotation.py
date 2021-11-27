from utils.torch_util import freeze

MAX_TRAIN_SAMPLES_PER_EPOCH = 2000
MAX_VAL_SAMPLES = 200

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', default=1)
parser.add_argument('hyper_config', help="Hyperparams config python file.")
parser.add_argument('ldd_config', help='Loss and diag dict config python file')
# parser.add_argument('loss_str', type=str, help="Loss str describing what loss fnx to use")
parser.add_argument('--resume_pkl', type=str, help="Checkpoint to resume from")
parser.add_argument('--load_optim', action='store_true')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--kld_weight', type=float, default=.00001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--checkpoint_freq', type=int, default=1)
parser.add_argument('--evaluation_freq', type=int, default=1, help="Evaluate every n epochs")
parser.add_argument('--augment_mesh_freq', type=int, default=100)

parser.add_argument('--max_partitions', type=int, default=20, help="Max number of batches to train each epoch")
parser.add_argument('--max_evaluation_episodes', type=int, default=10)

parser.add_argument('--enable_scheduler', action='store_true')
parser.add_argument('--num_warmup_steps', type=int, default=500, help='Number of steps to warmup with the LR scheduler')
parser.add_argument('--num_training_steps', type=int, default=2000, help='Number of steps before LR rate converges to 0,'
                                                                         'after warming up')
parser.add_argument('--clip', type=int, default=1, help='Gradient clipping')
parser.add_argument('--seed', type=int, default=0, help='Torch and Numpy random seed')
parser.add_argument('--test_model_load_reproducibility', action='store_true', help="Serialize and load model and check that "
                                                                              "evaluation performance is the same")
parser.add_argument('--num_points', type=int, default=150, help="Num points for FPS sampling")
parser.add_argument('--num_fps_samples', type=int, default=1, help="Num samples for FPS sampling")
parser.add_argument('--resume_initconfig', type=str, help="Initconfig to resume from")
parser.add_argument('--rot_mag_bound', type=float, help="How much to bound the rotation magnitude", default=2*3.14159)

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser.add_argument('--rot_aug', default="xyz", type=none_or_str)
parser.add_argument('--scale_aug', default="xyz", help="What type of scale aug to use")
parser.add_argument('--shear_aug', type=str, default="xy")

parser.add_argument('--log_gradients', action='store_true', help="Log gradients")
parser.add_argument('--max_z_scale', type=float, default=2.0)
parser.add_argument('--min_z_scale', type=float, default=0.5)
parser.add_argument('--max_shear', type=float, default=0.5)
parser.add_argument('--detect_anomaly', action='store_true')
parser.add_argument('--freeze_threshold', type=float, default=0.05)
parser.add_argument('--freeze_encoder', action='store_true', help="Freeze encoder if reconstruction error falls below"
                                                                  "freeze threshold")
parser.add_argument('--freeze_decoder', action='store_true', help="Freeze decoder")
parser.add_argument('--gpu0', type=int, default=0, help="GPU ID to use for multi GPU 0")
parser.add_argument('--gpu1', type=int, default=1, help="GPU ID to use for multi GPU 1")

parser.add_argument('--train_dset_path', type=str)
parser.add_argument('--val_dset_path', type=str)

parser.add_argument('--stats_json', type=str)

parser.add_argument('--center_fps_pc', action='store_true', help='Center FPS')

parser.add_argument('--no_linear_search', action='store_false', help='Use linear search for the plane label generation')

parser.add_argument('--threshold_frac',type=float, default=.02, help='Fraction of points to use for plane label generation')
parser.add_argument('--max_frac_threshold', type=float, default=.1)
parser.add_argument('--dont_make_btb', action='store_true')
parser.add_argument('--randomize_z_canonical', action='store_true')
parser.add_argument('--further_downsample_frac', default=None)
args = parser.parse_args()
import torch
torch.random.manual_seed(args.seed)

import numpy
numpy.random.seed(args.seed)

import random
random.seed(args.seed)

from utils import train_util, misc_util, loss_util
# from supervised_training.utils.misc_util import *
# from supervised_training.utils.loss_util import *
import wandb
import subprocess
# from supervised_training.optim.hugging_face_optimization import get_linear_schedule_with_warmup
import json

from torch.utils.data import DataLoader

import os
from data_generation.dataset import PointcloudDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(("utf-8")).split("\n")[0]

if args.detect_anomaly:
    torch.autograd.set_detect_anomaly(True)


wandb.init(project="python_oid_prediction", tags=["classifier"])
# wandb.config['git_id'] = git_hash
wandb.config['run_dir'] = wandb.run.dir
wandb.save(args.hyper_config)

config = misc_util.load_hyperconfig_from_filepath(args.hyper_config)

wandb.config.update(config)

wandb.config.update(vars(args))

models_dict = train_util.model_creator(config=config,
                            device_id=args.device_id)

model = next(iter(models_dict.items()))[1]

if hasattr(model, "multi_gpu") and model.multi_gpu:
    model.gpu_0 = torch.device(f"cuda:{args.gpu0}")
    model.gpu_1 = torch.device(f"cuda:{args.gpu1}")

if args.resume_pkl:
    pkl = torch.load(args.resume_pkl)
    random_sd = model.state_dict()
    random_sd.update(pkl['model'])
    model.load_state_dict(random_sd)


MODEL_DEVICE = next(model.parameters()).device

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr)

if args.resume_pkl and args.load_optim:
    pkl = torch.load(args.resume_pkl)
    optimizer.load_state_dict(pkl['opt'])

num_epochs = 100000

# if args.enable_scheduler:
#     scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.num_training_steps)

forward_pass_dict = dict()


total_iteration = 0

if args.log_gradients:
    wandb.watch(model, log_freq=1)

with open(args.stats_json, "r") as fp:
    stats_dic = json.load(fp)

below_freeze_threshold_count = 0

train_dset = PointcloudDataset(args.train_dset_path,
                               stats_dic=stats_dic,
                               center_fps_pc=args.center_fps_pc,
                               linear_search=args.no_linear_search,
                               threshold_frac=args.threshold_frac,
                               max_frac_threshold=args.max_frac_threshold,
                               rot_aug=args.rot_aug,
                               dont_make_btb=args.dont_make_btb,
                               randomize_z_canonical=args.randomize_z_canonical,
                               further_downsample_frac=args.further_downsample_frac)
val_dset = PointcloudDataset(args.val_dset_path,
                            stats_dic=stats_dic,
                             center_fps_pc=args.center_fps_pc,
                             linear_search=args.no_linear_search,
                             threshold_frac=args.threshold_frac,
                             max_frac_threshold=args.max_frac_threshold,
                             rot_aug=args.rot_aug,
                             dont_make_btb=args.dont_make_btb,
                             randomize_z_canonical=args.randomize_z_canonical,
                             further_downsample_frac=args.further_downsample_frac)
train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=args.batch_size, drop_last=True, shuffle=True,
                           num_workers=0)
val_dloader = DataLoader(val_dset, pin_memory=True, batch_size=args.batch_size, drop_last=True, shuffle=True)

get_loss_and_diag_dict = misc_util.load_ldd_function_from_filepath(args.ldd_config)

for epoch in range(num_epochs):
    print(f"ln73 Epoch {epoch}")

    if epoch % args.evaluation_freq == 0 and epoch > 0:
        model.eval()
        for batch_ndx, batch in enumerate(val_dloader):
            if batch_ndx * args.batch_size > MAX_VAL_SAMPLES:
                break
            optimizer.zero_grad()
            print("ln295 val forward")

            torch.cuda.empty_cache()

            with torch.no_grad():
                predictions = model(batch)
                val_loss, val_diag_dict = get_loss_and_diag_dict(predictions, batch,
                                                                 increment_iteration=False,
                                                                 loss_params=config['loss_params'])
            wandb_dict = {"val/total_loss": val_loss.data.cpu().numpy(),
                              "val/total_iteration": total_iteration,
                              "epoch": epoch}
            wandb_dict.update(**{f"val/{k}": v for k, v in val_diag_dict.items()})

            # if "cvae_bce_cat" in args.loss_str:
            #     wandb_dict.update(current_temp=models_dict['surface_classifier'].current_temp,
            #                       )

            wandb.log(wandb_dict, step=total_iteration)
        model.train()
    if epoch % args.checkpoint_freq == 0:
        dic = dict(model=model.state_dict(),
                   opt=optimizer.state_dict(),
                   )
        # if args.enable_scheduler:
        #     dic['scheduler_get_lr'] = scheduler.get_lr()
        torch.save(dic, os.path.join(wandb.run.dir, f"{wandb.run.name}_checkpoint{epoch}"))

        print("ln177 checkpoint path")
        print(os.path.join(wandb.run.dir, f"checkpoint{epoch}"))

    # for iter_ in range(100):
    for batch_ndx, batch in enumerate(train_dloader):
        if batch_ndx * args.batch_size > MAX_TRAIN_SAMPLES_PER_EPOCH:
            break

        if args.freeze_encoder:
            freeze(model.pointcloud_encoder)
            freeze(model.fc_embedding2z)

        if args.freeze_decoder:
            freeze(model.pointcloud_decoder)

        predictions = model(batch)

        optimizer.zero_grad()

        loss, diag_dict = get_loss_and_diag_dict(predictions, batch,
                                             loss_params=config['loss_params'])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        # if args.enable_scheduler:
        #     scheduler.step()

        total_iteration += 1

        wandb_dict = dict(total_loss=loss.data.cpu().numpy(),
                          total_iteration=total_iteration,
                          epoch=epoch)

        wandb_dict.update(**diag_dict)

        """
        Custom Diagnostics for Various Models
        """
        # if "cvae" in args.loss_str:
        #     if "prior_kld" in config['loss_params']['kl_type']:
        #         wandb_dict.update(current_temp=model.current_temp)
        #
        #     if "cvae_bce_cat" in args.loss_str:
        #         wandb_dict.update(current_temp=models_dict['surface_classifier'].current_temp)
        #
        #     if "maf" in args.loss_str:
        #         wandb_dict.update(prior_flow_log_prob=
        #                           wandb.Histogram(predictions['prior'][0].data.cpu().numpy()))
        #
        #     for k, v in predictions['prior'].items():
        #         wandb_dict[k] = wandb.Histogram(v.data.cpu().numpy())
        #
        #     for k, v in predictions['encoder'].items():
        #         wandb_dict[k] = wandb.Histogram(v.data.cpu().numpy())

        for k, v in wandb_dict.items():
            if isinstance(v, torch.Tensor):
                wandb_dict[k] = v.data.cpu().numpy()
        wandb.log(wandb_dict, step=total_iteration)
