# New Features
- Contact plane registration implementation
- CVAE models, including implementation of Mixture of Gaussian prior
- CVAE loss functions

# Training
1. 

# Generating data
## SO(3) augmentation

```
python3 data_generation/1_generate_pointclouds_v2.py parts/main/bandu_train/ nov27 --no_table --no_simulate
```

## Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl
```


# Credits

- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/FlyingGiraffe/vnn
- https://github.com/AntixK/PyTorch-VAE/tree/master/models