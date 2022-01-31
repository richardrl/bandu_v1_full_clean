# New Features
- Contact plane registration implementation
- CVAE models, including implementation of Mixture of Gaussian prior
- CVAE loss functions
- Highly-parallelized data generation

# Training
```
python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py 
configs/losses/cvae_btb_loss_config.py out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples 
out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples --stats_json=out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples/rr_pn_stats.json

python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/losses/cvae_btb_loss_config.py out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples out/canonical_pointclouds/jan8_val/fps_randomizenoiseTrue_numfps10_samples --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json

python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/losses/cvae_btb_loss_config.py out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples out/canonical_pointclouds/jan8_val/fps_randomizenoiseTrue_numfps10_samples --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json --batch_size=32 --gpu0=8 --gpu1=9 --device_id=9


python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/losses/cvae_btb_loss_config.py out/datasets/bandu_train/jan18_train/voxelized_samples out/datasets/bandu_val/jan18_val/voxelized_samples --stats_json=out/datasets/bandu_train/jan18_train/voxelized_samples/rr_pn_stats.json --batch_size=32 --gpu0=6 --gpu1=7 --device_id=7 --threshold_frac=.06 --max_frac_threshold=.2
```

# Training loss visualization
We use Wandb to visualize losses during training. 
You are welcome to roll your own visualization, just comment out the lines involving wandb.

# Generating data
### SO(3) augmentation only, no dropping from table

```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/bandu_train/ test --no_table --no_simulate

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/bandu_train/test/canonical_pointcloud_samples 2 1
 
# make sure to fill in the settings at the top of the file!
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples 0
```

### Standard shape + SO(3) augmentation, dropping onto table, used in training for our paper

10 samples in simulator per object

10 farthest point samples for each of the above samples

Train
```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/bandu_train jan18_train --num_samples=10

python3 data_generation/2_generate_fps_pointclouds_2.py out/datasets/bandu_train/jan18_train/canonical_pointcloud_samples
 
# make sure to fill in the settings at the top of the file!
python3 data_generation/calculate_stats_json.py out/datasets/bandu_train/jan18_train/voxelized_samples 0
```

### Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/sundisk/4.pkl

```

# Loading model and evaluating samples

## Real samples

spring-plasma-2020 (paper model)
```
python3 test_single_real_sample.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py out/spring-plasma-2020_checkpoint240 --stats_json=out/datasets/bandu_train/jan18_train/voxelized_samples/rr_pn_stats.json /data/pulkitag/models/rli14/realsense_docker/out/samples/01-27-2022_00:28:17_JBlock_real2sim.torch
```

## Sim samples
```
python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py wandb/run-20220110_003726-cbdk34il/files/vocal-fire-24_checkpoint416 out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json

python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py wandb/run-20220110_003726-cbdk34il/files/vocal-fire-24_checkpoint416 out/canonical_pointclouds/jan8_val/fps_randomizenoiseTrue_numfps10_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json

# spring-plasma-2020 (paper model)
python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py out/spring-plasma-2020_checkpoint240 out/canonical_pointclouds/jan8_test/fps_randomizenoiseTrue_numfps2_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json
```

# Evaluating stacking capability of trained policy in gym environment

TODO: get it working without block base

```
python3 environment_evaluation/1_visualize_trained_eval_in_environment_rollout.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/envs/bandu_train_cameraon_verticalurdfs_sequential_cmvscvaecm.py --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json --results_dir=out/env_eval/  --sc_checkpoint=out/spring-plasma-2020_checkpoint240 --block_base
```

vocal-fire-24 (new codebase model)
Change the results_dir, device ID and seed when parallelizing this evaluation over multiple GPUs
```
python3 environment_evaluation/1_visualize_trained_eval_in_environment_rollout.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/envs/bandu_train_cameraon_verticalurdfs_sequential_cmvscvaecm.py --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json --results_dir=out/vocal-fire-24/env_eval_visiongpu51_gpu6/  --sc_checkpoint=wandb/run-20220110_003726-cbdk34il/files/vocal-fire-24_checkpoint600 --block_base --device_id=6 --seed=516

```

```

cd environment_evaluation 

python3 2_calculate_success_rate_2block.py ../parts/urdfs/main/engmikedset/ ../out/ 50

python3 2_calculate_success_rate_2block.py ../parts/urdfs/main/engmikedset/ ../out/vocal-fire-24/ 50

```

Fill out the arguments in make latex table, and run:

```
python3 make_latex_table.py

```

# Real2Sim experiments
```
python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/engmikedset jan26_real2sim --num_samples=1 --pb_loop --manual_pose --manually_choose_urdf

```


# Docker 
To write to files in the mounted volume, make an "out" folder with permissions 777.

We need cuda10 to visualize. cuda11 to run on 3090 gtx cards.

# Processing depth camera in sim
1. Get raw, large pointcloud from pybullet cameras
2. Voxel-based downsampling
3. Uniform downsampling to 2048 (we can handle variable sized pointclouds, but this is done to ensure we stay within the computational limits)

# Debugging issues
- If get_bti_from_rotated fails increase max_frac_threshold, we might just not be finding enough points

# Credits

- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/FlyingGiraffe/vnn
- https://github.com/AntixK/PyTorch-VAE/tree/master/models