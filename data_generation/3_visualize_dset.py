from data_generation.dataset import PointcloudDataset
import open3d as o3d
from utils import *
import sys
import numpy as np

# load dset
pcd_dset = PointcloudDataset(sys.argv[1],
                             center_fps_pc=False,
                             threshold_frac=.02)
                             # max_z_scale=.5,
                             # min_z_scale=.49)
                             # shear_aug=None,
                             # scale_aug=None)



sample_idx = np.random.randint(0, pcd_dset.__len__())

# load sample
sample = pcd_dset.__getitem__(3)

# visualize labels
# vpcd = vis_util.make_point_cloud_o3d(sample['rotated_pointcloud'][0], color=[0.,0.,0.])
vpcd = vis_util.make_point_cloud_o3d(sample['rotated_pointcloud'][0],
                                     color=vis_util.make_colors(sample['bottom_thresholded_boolean']))

# load sample
# sample = pcd_dset.__getitem__(1)
#
# # visualize
# vpcd2 = vis_util.make_point_cloud_o3d(sample['rotated_pointcloud'][0] + np.array([0., 0., 0.5]),
#                                       color=make_colors(sample['bottom_thresholded_boolean']))

vpcd2 = vis_util.make_point_cloud_o3d(sample['canonical_pointcloud'] + np.array([0., 0., 0.5]),
                                      color=np.array([0., 1., 0.]))
# import pdb
# pdb.set_trace()
o3d.visualization.draw_geometries([vpcd,
                                   vpcd2,
                                   o3d.geometry.TriangleMesh.create_coordinate_frame(.06, [0, 0, 0])])
                                   # o3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, -.5])])