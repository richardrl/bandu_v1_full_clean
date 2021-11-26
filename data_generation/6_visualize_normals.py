import open3d as o3d
import os
from pathlib import Path
import json
import numpy as np
import copy
from bandu.utils import bandu_util

# load STL
mesh = o3d.io.read_triangle_mesh("/home/richard/improbable/spinningup/parts/stls/main/engmikedset/Pyramid_W_Eye.stl")

mesh_after = copy.deepcopy(mesh)
mesh_after.paint_uniform_color(np.array([0., 1., 0.]))

urdf_path = "/home/richard/improbable/spinningup/parts/urdfs/main/engmikedset/Pyramid_W_Eye/Pyramid_W_Eye.urdf"

# load associated normals
dir_ = os.path.dirname(urdf_path)
config_path = Path(dir_) / "extra_config"
with open(str(config_path), "r") as fp:
    jd = json.load(fp)

# for each normal, orient the STL
for nm_list in jd['normals']:
    normal = np.array(nm_list)
    print("ln21 working normal...")
    print(normal)

    rotmat = bandu_util.get_rotation_matrix_between_vecs([0, 0, -1], normal)

    print("ln30 rotmat")
    print(rotmat)
    print(np.linalg.det(rotmat))
    mesh_after.rotate(rotmat, np.zeros(3))

    mesh_after.translate(np.array([0., 0., 0.2]))

    o3d.visualization.draw_geometries([mesh,
                                       mesh_after,
                                       o3d.geometry.TriangleMesh.create_coordinate_frame(.03, [0., 0., 0.2])])

    mesh_after.translate(np.array([0., 0., -0.2]))
    mesh_after.rotate(np.linalg.inv(rotmat), np.zeros(3))