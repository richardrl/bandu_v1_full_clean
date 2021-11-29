import numpy as np
import torch
import open3d as o3d


def make_point_cloud_o3d(points, color):
    if isinstance(points, torch.Tensor):
        points = points.data.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    color = np.array(color)
    if len(color.shape) == 1:
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    else:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_big_cloud(points, color, radius=.005):
    from bandu.utils.bandu_util import create_sphere
    spheres = []
    for pt in points:
        sphere = create_sphere(pt, color=color, radius=radius)
        spheres.append(sphere)
    return spheres


def text_3d(text, pos, direction=None, degree=0.0, font='Ubuntu-L.ttf', density=100, font_size=4):
    import numpy as np
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    text = str(text)
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

import open3d
from utils import transform_util


def create_arrow(vec, color, vis=None, vec_len=None, scale=.06, radius=.12, position=(0,0,0),
                 object_com=None):
    """
    Creates an error, where the arrow size is based on the vector magnitude.
    :param vec:
    :param color:
    :param vis:
    :param scale:
    :param position:
    :return:
    """
    if vec_len is None:
        vec_len = (np.linalg.norm(vec))

    mesh_arrow = open3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len * scale,
        cone_radius=0.06 * vec_len * radius,
        cylinder_height=0.8 * vec_len * scale,
        cylinder_radius=0.04 * vec_len * radius
    )

    # the default arrow points straightup
    # therefore, we find the rotation that takes us from this arrow to our target vec, "vec"

    if object_com is not None:
        vec_endpoint = position + vec
        neg_vec_endpoint = position - vec
        if np.linalg.norm(vec_endpoint - object_com) < np.linalg.norm(neg_vec_endpoint - object_com):
            vec = -vec

    # print("ln554 create_arrow")
    # print(vec)
    rot_mat = transform_util.get_rotation_matrix_between_vecs(vec, [0, 0, 1])
    print(rot_mat)
    mesh_arrow.rotate(rot_mat, center=np.array([0,0,0]))

    H = np.eye(4)
    H[:3, 3] = position
    mesh_arrow.transform(H)

    mesh_arrow.paint_uniform_color(color)
    return mesh_arrow