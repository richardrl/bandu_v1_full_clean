import torch
import os
import json
import tqdm
from deco import *
import sys
from torch.utils.data import DataLoader
from data_generation.dataset import PointcloudDataset

from scipy.spatial.transform import Rotation as R

# ic_path = "out/canonical_pointclouds/combined1/normals_test/initconfig"
"""
Settings
"""
train_dset_samples_dir = sys.argv[1]
use_normals = int(sys.argv[2])
batch_size = 34
threshold_frac = .06

# num_epochs = 100

# train_pcs = torch.load(ic_path)['pcs']

# for use with CVAE Seg
# encoder and decoder
device = torch.device("cuda:0")
rotated_pc_sum_vector = torch.zeros(3).to(device)
rotated_pc_var_sum_vector = torch.zeros(3).to(device)

rotated_normals_sum_vector = torch.zeros(3).to(device)
rotated_normals_var_sum_vector = torch.zeros(3).to(device)

canonical_pc_sum_vector = torch.zeros(3).to(device)
canonical_pc_var_sum_vector = torch.zeros(3).to(device)

rotated_quat_inv_sum_vector = torch.zeros(4).to(device)
rotated_quat_inv_var_sum_vector = torch.zeros(4).to(device)

total_num_points = 0

train_dset = PointcloudDataset(train_dset_samples_dir,
                               stats_dic=None,
                               threshold_frac=threshold_frac,
                               max_frac_threshold=.2)
train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=batch_size, drop_last=True, shuffle=True,
                           num_workers=0)

print("ln37 total dset samples")
print(train_dset.__len__())

print("ln40 total batches in dataloader")
print(train_dloader.__len__())

total_samples = 0

for phase in ['mean', 'var']:
    # for epoch in tqdm.tqdm(range(num_epochs)):
    for batch_ndx, batch in enumerate(tqdm.tqdm(train_dloader)):
        print("Batch index")
        print(batch_ndx)
    # for epoch in range(num_epochs):
    #     batch = process_batch_relativerotation(train_pcs,
    #                                            batch_size,
    #                                            scale_aug="xyz",
    #                                            rot_aug="z",
    #                                            rot_mag_bound=2*3.14159,
    #                                            max_z_scale=2.0,
    #                                            min_z_scale=.5,
    #                                            max_shear=.5,
    #                                            shear_aug="xy",
    #                                            model_device=device,
    #                                            use_normals=use_normals)
        if phase == "mean":
            rotated_pc_sum_vector += batch['rotated_pointcloud'].reshape(-1, 3).sum(dim=0).to(device)
            canonical_pc_sum_vector += batch['canonical_pointcloud'].reshape(-1, 3).sum(dim=0).to(device)

            rqi = torch.Tensor(R.from_quat(batch['rotated_quat']).inv().as_quat()).sum(dim=0).to(device)
            rotated_quat_inv_sum_vector += rqi

            total_samples += batch['rotated_quat'].shape[0]
            if use_normals:
                rotated_normals_sum_vector += batch['rotated_normals'].reshape(-1, 3).sum(dim=0).to(device)

            print("ln34 rotated_pc_sum_vector")
            print(rotated_pc_sum_vector)
            total_num_points += batch['rotated_pointcloud'].reshape(-1, 3).shape[0]
            print("ln37 total number of POINTS")
            print(total_num_points)
        else:
            # phase == "rotated_pc_var"
            rotated_pc_mean = (rotated_pc_sum_vector / total_num_points).to(device)

            rotated_pc_var_sum_vector += ((batch['rotated_pointcloud'].reshape(-1, 3).to(device) -
                                           rotated_pc_mean) ** 2).sum(dim=0).to(device)

            rotated_quat_inv_mean = (rotated_quat_inv_sum_vector / total_samples).to(device)


            rotated_quat_inv_var_sum_vector += ((torch.Tensor(R.from_quat(batch['rotated_quat']).inv().as_quat()).to(device) - \
                                               rotated_quat_inv_mean.unsqueeze(0))**2).sum(dim=0).to(device)

            canonical_pc_mean = (canonical_pc_sum_vector / total_num_points).to(device)
            canonical_pc_var_sum_vector += ((batch['canonical_pointcloud'].reshape(-1, 3).to(device) -
                                             canonical_pc_mean) ** 2).sum(dim=0).to(device)

            if use_normals:
                rotated_normals_mean = rotated_normals_sum_vector / total_num_points
                rotated_normals_var_sum_vector += ((batch['rotated_normals'].reshape(-1, 3) -
                                                    rotated_normals_mean) ** 2).sum(dim=0).to(device)

rotated_pc_mean = rotated_pc_sum_vector / total_num_points
rotated_pc_var = rotated_pc_var_sum_vector / (total_num_points - 1)

rotated_quat_inv_mean = rotated_quat_inv_sum_vector / total_samples
rotated_quat_inv_var = rotated_quat_inv_var_sum_vector / (total_samples - 1)

canonical_pc_mean = canonical_pc_sum_vector / total_num_points
canonical_pc_var = canonical_pc_var_sum_vector / (total_num_points - 1)

if use_normals:
    rotated_normals_mean = rotated_normals_sum_vector / total_num_points
    rotated_normals_var = rotated_normals_var_sum_vector / (total_num_points - 1)
    assert torch.all(rotated_normals_var > 0)

assert torch.all(rotated_pc_var > 0)
assert torch.all(canonical_pc_var > 0)

dic = dict(rotated_pointcloud_mean=rotated_pc_mean.data.cpu().numpy().tolist(),
           rotated_pointcloud_var=rotated_pc_var.data.cpu().numpy().tolist(),
           canonical_pointcloud_mean=canonical_pc_mean.data.cpu().numpy().tolist(),
           canonical_pointcloud_var=canonical_pc_var.data.cpu().numpy().tolist(),
           rotated_quat_inv_mean=rotated_quat_inv_mean.data.cpu().numpy().tolist(),
           rotated_quat_inv_var=rotated_quat_inv_var.data.cpu().numpy().tolist()
)

if use_normals:
    dic.update(rotated_normals_mean=rotated_normals_mean.data.cpu().numpy().tolist(),
               rotated_normals_var=rotated_normals_var.data.cpu().numpy().tolist())

print("ln96 rotated pc stats")
print(rotated_pc_mean)
print(rotated_pc_var)

print("ln100 canonical pc stats")
print(canonical_pc_mean)
print(canonical_pc_var)

with open(os.path.join(train_dset_samples_dir, "rr_pn_stats.json"), "w") as fp:
    json.dump(dic, fp)