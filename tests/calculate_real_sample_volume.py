import torch
import sys
import numpy as np
import glob


# sample_pkl = torch.load(sys.argv[1])

sample_pkls = glob.glob(f"{sys.argv[1]}/*")

sum_lengths=  np.zeros(3)
sum_volume = 0


for sample_pkl_path in sample_pkls:
    sample_pkl = torch.load(sample_pkl_path)
    lengths = np.max(sample_pkl['points'], axis=0) - np.min(sample_pkl['points'], axis=0)
    sum_lengths += lengths

    sum_volume += np.prod(lengths)

# print("lengths")
# print(lengths)
#
# print("volume")
# print(np.prod(lengths))

print("avg lengths")
print(sum_lengths/len(sample_pkls))
print("avg volume")
print(sum_volume/len(sample_pkls))