import torch
import open3d
from bandu.utils import pb_util, vis_util, bandu_util
from supervised_training.utils.visualization_util import make_colors
from bandu.utils import color_util
import numpy as np
from supervised_training.utils import surface_util, pointcloud_util, visualization_util
import sys

dic = torch.load(sys.argv[1])

selected_object_id = 0
mc1 = make_colors(dic['btb'][selected_object_id].squeeze(-1),
                  background_color=color_util.MURKY_GREEN, surface_color=color_util.YELLOW)
mc2 = make_colors(dic['btb'][selected_object_id].squeeze(-1))

# vis PCA bounding box, face, and normal...
relmat, oriented_source_vector = surface_util.get_relative_rotation_from_obb(dic['rotated_pointcloud'],
                                            vis_o3d=True,
                                            gen_antiparallel_rotation=False,
                                                                             ret_oriented_source_vector=True)

from scipy.spatial.transform import Rotation as R
"""
IPDF start
"""
from imports.ipdf import models as ipdf_models
from imports.ipdf import visualization
from supervised_training.models.dgcnn_cls import DGCNNCls
device = torch.device(f"cuda:0")

ipdf = ipdf_models.ImplicitPDF(feature_size=256, num_layers=2).to(device)

ipdf_checkpoint = "/home/richard/Downloads/bandu_ipdf_ckpt_l2lr1em4_e40"

model = DGCNNCls(num_class=256).to(device)

pkl = torch.load(ipdf_checkpoint, map_location=device)
model.load_state_dict(pkl['model'])
ipdf.load_state_dict(pkl['ipdf'])

model.eval()
ipdf.eval()

# Make dummy point cloud input.
point_cloud = torch.rand(1, 1, 2048, 3)

models_dict = dict()
models_dict['model'] = model
models_dict['ipdf'] = ipdf

with torch.no_grad():
    feature = models_dict['model'](dic['rotated_pointcloud'].unsqueeze(0).permute(0, 2, 1))
    # Recursion level 4 amounts to ~300k samples.
    queries = ipdf_models.generate_queries(
        num_queries=4,
        mode='grid',
        rotate_to=torch.eye(3, device=device)[None])
    pdf, pmf = models_dict['ipdf'].compute_pdf(feature, queries)

    # If we have to output a single rotation, this is it.
    # TODO: we could run gradient ascent here to improve accuracy.
    relative_rotmat = queries[0][pdf.argmax(axis=-1)][0].data.cpu().numpy()


    # apply the rel_rot so we are in canonical pose
    # draw the vector from COM down to [0, 0, min_height_point]
    # add a normal from that point to [0, 0, -1]
    # invert the rel_rot

    canonical_pc = R.from_matrix(relative_rotmat).apply(dic['rotated_pointcloud'].data.cpu().numpy())

    min_height = np.min(canonical_pc[np.nonzero(np.linalg.norm(canonical_pc[:, :2], axis=-1) < .05)][:, -1])

    normal_base = np.array([0., 0., min_height])
    normal = np.array([0., 0., -1])

    # rotate back
    normal_base_r = R.from_matrix(relative_rotmat).inv().apply(normal_base)
    normal_r = R.from_matrix(relative_rotmat).inv().apply(normal)


    open3d.visualization.draw_geometries([
        vis_util.make_point_cloud_o3d(dic['rotated_pointcloud'],
                                      color=np.zeros(3)),
        bandu_util.create_arrow(normal_r, [0., 0., .5],
                                position=normal_base_r,
                                object_com=np.zeros(3)),
        open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0., 0., 0.])])


"""
IPDF end
"""

open3d.visualization.draw_geometries([
    vis_util.make_point_cloud_o3d(dic['rotated_pointcloud'],
                                  color=dic['mc1']),
    # vis_util.make_point_cloud_o3d(dic['canonical_pc'] + np.array([0.,0.,.5]),
    #                               color=dic['mc2']),
    # bandu_util.create_arrow(dic['plane_model'][:3], [1., 0., 0.]),
    bandu_util.create_arrow(np.array(dic['plane_model'][:3]), [0., 0., .5],
                            position=np.array(dic['box_centroid']),
                            object_com=np.zeros(3)),
    surface_util.gen_surface_box(dic['plane_model'], ret_centroid=True, color=[0., 0., .5])[0], # plane model based on the rotated pc
    open3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0., 0., 0.])])

