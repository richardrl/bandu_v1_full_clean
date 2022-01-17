from utils import vis_util
import open3d as o3d
import torch
import sys

sample_pkl_path = sys.argv[1]

sample = torch.load(sample_pkl_path)

# visualize
vpcd = vis_util.make_point_cloud_o3d(sample['original_rotated_centered_pointcloud'], color=[0.,0.,0.])
# vpcd = vis_util.make_point_cloud_o3d(sample['rotated_pointcloud'][0],
#                                      color=make_colors(sample['bottom_thresholded_boolean']))

# visualize
# vpcd2 = vis_util.make_point_cloud_o3d(sample['rotated_pointcloud'][0] + np.array([0., 0., 0.5]),
#                                       color=make_colors(sample['bottom_thresholded_boolean']))


o3d.visualization.draw_geometries([vpcd,
                                   # vpcd2,
                                   o3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, 0])])
                                   # o3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0, 0, -.5])])
