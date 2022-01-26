from data_generation.sim_dataset import PybulletPointcloudDataset
import sys
import numpy as np
# from torch.data import Dataloader

train_dset_samples_dir = sys.argv[1]
# use_normals = int(sys.argv[2])
threshold_frac = .06
max_frac_threshold = .2

train_dset = PybulletPointcloudDataset(train_dset_samples_dir,
                               stats_dic=None,
                               threshold_frac=threshold_frac,
                               max_frac_threshold=max_frac_threshold)
# train_dloader = DataLoader(train_dset, pin_memory=True, batch_size=batch_size, drop_last=True, shuffle=True,
#                            num_workers=0)


print(f"About to iterate over dataset of len {len(train_dset)}")
avg_lengths = np.zeros(3)
avg_volume = 0

for idx in range(len(train_dset)):
    print(f"idx {idx}")
    lengths = np.max(train_dset.__getitem__(idx)['canonical_pointcloud'], axis=0) - np.min(train_dset.__getitem__(idx)['canonical_pointcloud'], axis=0)
    print(lengths)
    print(np.prod(lengths))

    avg_lengths += lengths
    avg_volume += np.prod(lengths)


print("avg lengths")
print(avg_lengths/len(train_dset))

print('avg volume')
print(avg_volume/len(train_dset))