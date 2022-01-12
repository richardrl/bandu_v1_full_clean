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
```

# Training loss visualization
We use Wandb to visualize losses during training. 
You are welcome to roll your own visualization, just comment out the lines involving wandb.

# Generating data
### SO(3) augmentation only

```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/bandu_train/ test --no_table --no_simulate

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/bandu_train/test/canonical_pointcloud_samples 2 1
 
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples 0
```

### Standard augmentation used in training for our paper

10 samples in simulator per object

10 farthest point samples for each of the above samples

Train
```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/bandu_train/ jan8_train --num_samples=10

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/jan8_train/canonical_pointcloud_samples 10 1
 
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples 0
```

Val 
```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/bandu_val/ jan8_val --num_samples=10

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/jan8_val/canonical_pointcloud_samples 10 1
 
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/jan8_val/fps_randomizenoiseTrue_numfps10_samples 0
```

Test (engmikedset)
```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/urdfs/main/engmikedset/ jan8_test --num_samples=2

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/jan8_test/canonical_pointcloud_samples 2 1
 
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/jan8_test/fps_randomizenoiseTrue_numfps10_samples 0
```

### Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/sundisk/4.pkl

```

# Loading model and evaluating samples

## Real samples

```
python3 test_single_real_sample.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py out/spring-plasma-2020_checkpoint240 --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json /data/pulkitag/models/rli14/realsense_docker/out/samples/01-09-2022_00:36:11_Cross.torch

```

## Sim samples
```
python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py wandb/run-20220110_003726-cbdk34il/files/vocal-fire-24_checkpoint416 out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json

python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py wandb/run-20220110_003726-cbdk34il/files/vocal-fire-24_checkpoint416 out/canonical_pointclouds/jan8_val/fps_randomizenoiseTrue_numfps10_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json

python3 visualize_simulated_classified_surface_points.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py out/spring-plasma-2020_checkpoint240 out/canonical_pointclouds/jan8_test/fps_randomizenoiseTrue_numfps2_samples  --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json
```

# Evaluating stacking capability of trained policy in gym environment
```
python3 environment_evaluation/visualize_trained_eval_in_environment_rollout.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/envs/bandu_train_cameraon_verticalurdfs_sequential_cmvscvaecm.py --stats_json=out/canonical_pointclouds/jan8_train/fps_randomizenoiseTrue_numfps10_samples/rr_pn_stats.json --results_dir=out/env_eval/  --sc_checkpoint=out/spring-plasma-2020_checkpoint240

```

# Docker 
To write to files in the mounted volume, make an "out" folder with permissions 777.

We need cuda10 to visualize. cuda11 to run on 3090 gtx cards.

# Credits

- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/FlyingGiraffe/vnn
- https://github.com/AntixK/PyTorch-VAE/tree/master/models