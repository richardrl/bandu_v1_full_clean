# New Features
- Contact plane registration implementation
- CVAE models, including implementation of Mixture of Gaussian prior
- CVAE loss functions

# Training
```
python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py 
configs/losses/cvae_btb_loss_config.py out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples 
out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples --stats_json=out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples/rr_pn_stats.json

```

# Training loss visualization
We use Wandb to visualize losses during training. 
You are welcome to roll your own visualization, just comment out the lines involving wandb.

# Generating data
### SO(3) augmentation

```
cd bandu_v1_full_clean

python3 data_generation/1_generate_pointclouds_v2.py parts/main/bandu_train/ test --no_table --no_simulate

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/bandu_train/test/canonical_pointcloud_samples 2 1
 
python3 data_generation/calculate_stats_json.py out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples 0
```

### Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl
```

# Loading model and evaluating

```
python3 test_single_sample.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py /root/bandu_v1_full_clean/out/spring-plasma-2020_checkpoint240 --stats_json=out/canonical_pointclouds/bandu_train/test/fps_randomizenoiseTrue_numfps2_samples/rr_pn_stats.json

```

# Docker 
To write to files in the mounted volume, make an "out" folder with permissions 777.

# Credits

- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/FlyingGiraffe/vnn
- https://github.com/AntixK/PyTorch-VAE/tree/master/models