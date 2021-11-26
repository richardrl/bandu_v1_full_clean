# bandu_v1_full_clean


# Generating data
### SO(3) augmentation:

```
python3 data_generation/1_generate_pointclouds_v2.py parts/main/bandu_train/ test --no_table --no_simulate
```

# Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl
```