import argparse
import math
import os


# Remember, tf is installed with `pip install transformations`, not `pip install tf`
import random
from itertools import combinations
from pathlib import Path
# Bounding box lines where each element represeents the index of the point
import pybullet as p
import pybullet_data
import json

from bandu.utils import misc_util
from bandu.utils.color_util import RED, YELLOW, GREEN
from bandu.utils.data_util import random_z_axis_orientation
from bandu.utils.transform_util import get_rotation_matrix_between_vecs
from bandu.utils.vr_util import load_pr2_gripper, convert_windowspath_to_posixpath, \
    detect_vr_controller_and_load_plugin, detect_vr_controller_and_sync_pb_gripper
from bandu.config import BANDU_ROOT, bandu_logger

import urdfpy
import time

BBOX_EDGES_AS_POINT_IDX_PAIRS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 6],
    [1, 7],
    [2, 5],
    [2, 7],
    [3, 5],
    [3, 6],
    [4, 5],
    [4, 6],
    [4, 7],
]

EDGES_2D = [(0, 1), (1, 3), (3, 2), (2, 0)]

bbox_line_colors = [[0, 0, .5] for i in range(len(BBOX_EDGES_AS_POINT_IDX_PAIRS))]


# Util for util


def get_vector_from_point_pair_idxs(box_points, point_idx_pair, anchor_idx=None):
    list_copy = list(point_idx_pair)
    if anchor_idx is None:
        return box_points[list_copy[1]] - box_points[list_copy[0]]
    else:
        list_copy.remove(anchor_idx)
        return box_points[list_copy[0]] - box_points[anchor_idx]

def get_pair_contains_pairitem(iterable, pair_item):
    """
    Searches for a pair containing pairitem in the iterable.
    Once finding this pair, return the item that is NOT the pairitem.
    Args:
        iterable:
        pair_item:

    Returns:

    """
    for pair in iterable:
        if pair_item in pair:
            for item in pair:
                if item is not pair_item:
                    return item
    raise Exception


def get_vec_on_table(plane_model, xy):
    a, b, c, d = plane_model
    z = (-d - a * xy[0] - b * xy[1]) / c
    return np.asarray([xy[0], xy[1], z])


def get_normal(plane_model):
    a, b, c, d = plane_model
    return np.array([a, b, c])/np.linalg.norm(np.array([a, b, c]))


def get_table_aligned_obb(cluster_pts,
                          # table_normal,
                          # translation,
                          plane_model,
                          vis):
    import open3d
    a,b,c,d = plane_model
    table_normal = get_normal(plane_model)

    table2z_rot = get_rotation_matrix_between_vecs(np.array([0, 0, 1]), start_vec=table_normal)
    cluster_pts_z_aligned = np.dot(cluster_pts, table2z_rot.T)
    extentZ = np.max(cluster_pts_z_aligned[:, 2]) - (-d/c)

    corner_points_standard_2d, centerXY, extentX, extentY, eigenmatrixT = get_2d_obb(cluster_pts, vis)

    # Rotate around box around XY plane based on PCA results,
    # then rotate normal to align with the table
    # extent comes from the 2d PCA XY, plus a height
    if extentX >= extentY:
        XYrotation = get_rotation_matrix_between_vecs(np.append(eigenmatrixT[0], 0), np.asarray([1, 0, 0]))
    else:
        XYrotation = get_rotation_matrix_between_vecs(np.append(eigenmatrixT[1], 0), np.asarray([0, 1, 0]))

    XYZrot = np.dot(XYrotation, table2z_rot)
    center = get_vec_on_table(plane_model, centerXY) + get_normal(plane_model) * (extentZ/2)
    obb = open3d.geometry.OrientedBoundingBox(center=center,
                                              R=XYZrot,
                                              extent=[extentX, extentY, extentZ])

    # shortest_edge_vec, longest_edge_vec, bbox_lines = get_shortlong_and_bboxlines(np.asarray(obb.get_box_points()),
    dic = get_edge_dict(np.asarray(obb.get_box_points()),
                        bbox_line_colors,
                        all_faces_bbox_lines=BBOX_EDGES_AS_POINT_IDX_PAIRS)

    # bottom_surface_corner_points_3d = np.zeros((corner_points_standard_2d.shape[0], corner_points_standard_2d.shape[1] + 1))
    # bottom_surface_corner_points_3d[:, :2] = corner_points_standard_2d
    # bottom_surface_corner_points_3d[:, 2] = np.zeros_like(corner_points_standard_2d[:, 0])
    #
    # # Rotate 3d points from flat along xy plane to table plane
    # table_rot = get_rotation_between(table_normal, start_vec=np.array([0, 0, 1]))
    # bottom_surface_corner_points_3d = np.dot(bottom_surface_corner_points_3d, table_rot.T)
    # bottom_surface_corner_points_3d[2] += -d/c
    #
    # top_surface_corner_points_3d = bottom_surface_corner_points_3d + extentZ * table_normal/np.linalg.norm(table_normal)
    #
    # box_points = np.concatenate((
    #         bottom_surface_corner_points_3d,
    #         top_surface_corner_points_3d))
    # bbox_lines = EDGES_2D + [tuple([tup[0] + 4, tup[1]+4]) for tup in EDGES_2D]
    #
    # bbox_line_set = open3d.geometry.LineSet(
    #     points=open3d.utility.Vector3dVector(box_points),
    #     lines=open3d.utility.Vector2iVector(),
    # )
    # viz.add_geometry(bbox_line_set)

    dic['obb'] = obb
    dic['bbox_colored_lineset'] = get_bbox_colored_lineset(np.asarray(obb.get_box_points()),
                                       dic['parallel_edges_as_tuples'],
                                       top_face_shortest_edge=dic['top_face_shortest_edge_idx'])
    dic['longest_edge_vec'] = get_allfaces_longest_edge(np.asarray(obb.get_box_points()))

    # if np.linalg.norm(dic['top_face_longest_edge_vec']) != np.linalg.norm(dic['longest_edge_vec']):
    #     # Oblong shape lying flat on table
    #
    # else:
    #     # Oblong shape perpendicular to table
    return dic


def get_allfaces_longest_edge(box_points):
    all_faces_sorted_bbox_lines = map(sorted, BBOX_EDGES_AS_POINT_IDX_PAIRS)
    edges_as_vectors = map(lambda point_pair: get_vector_from_point_pair_idxs(box_points, point_pair), all_faces_sorted_bbox_lines)
    edge_lens = map(np.linalg.norm, edges_as_vectors)
    longest_edge_vector = sorted(zip(edge_lens, edges_as_vectors), key=lambda tup: tup[0])[-1]
    return longest_edge_vector


def get_frame_from_edge(longest_edge_points_idxs_tuple, box_points):
    """
    Gets unnormalized frame from an edge by 1) picking a point on the edge and 2) searching for the two other edges
    that share the same point to form a frame of 3 edges.
    :param edge:
    :return:
    """
    longest_point_pair, longest_idxs_pair = longest_edge_points_idxs_tuple
    anchor_idx = longest_idxs_pair[0]
    found_point_idx_pairs = []
    for point_idx_pair in BBOX_EDGES_AS_POINT_IDX_PAIRS:
        if anchor_idx in point_idx_pair:
            found_point_idx_pairs.append(point_idx_pair)
    found_point_idx_pairs.remove(longest_idxs_pair)

    # Make sure longest edge is always last so correspondences do what we want
    found_point_idx_pairs.append(longest_idxs_pair)

    # We should have 3 matching point idx pairs matching the longest idx pair, including itself
    assert len(found_point_idx_pairs) == 3

    # TODO: ensure that each vector is subtracting the anchor_idx idx... otherwise it won't be a frame

    def normalize(vec):
        return vec/(np.linalg.norm(vec))

    vectors = [get_vector_from_point_pair_idxs(box_points, idxs, anchor_idx=anchor_idx) for idxs in found_point_idx_pairs]
    vectors = sorted(vectors,
                     key=lambda x:np.linalg.norm(x))

    # bandu_logger.debug("Vectors")
    # bandu_logger.debug(vectors)
    # bandu_logger.debug("Vec lengths")
    # bandu_logger.debug([np.linalg.norm(vec) for vec in vectors])

    # -> [N, 3]
    return np.array(vectors)


def get_edge_dict(box_points):
    """

    :param box_points:
    :return: Dictionary of edge information
    """
    all_faces_sorted_bbox_lines = map(sorted, BBOX_EDGES_AS_POINT_IDX_PAIRS)

    # Get edges corresponding to each
    point_pairs = [np.array([box_points[point_idx_pair[0]], box_points[point_idx_pair[1]]]) for point_idx_pair in BBOX_EDGES_AS_POINT_IDX_PAIRS]
    zipped = zip(point_pairs,
                 BBOX_EDGES_AS_POINT_IDX_PAIRS)

    # For each tuple in the zipped, tup[0] gives the point pair, and we want to use
    # the distance between the points in the point pair as the sorting key
    longest_edge_points_idxs_tuple = sorted(zipped, key=lambda tup: np.linalg.norm(tup[0][0] - tup[0][1]))[-1]


    # Take 4 highest points as the top face of the rectangle
    assert type(box_points) == np.ndarray
    sorted_bp_indices = box_points[:, 2].argsort()[-4:]

    potential_edges = []
    for edge in all_faces_sorted_bbox_lines:
        if (edge[0] in sorted_bp_indices) and (edge[1] in sorted_bp_indices):
            potential_edges.append(tuple(edge))
    potential_edge_pairs = combinations(potential_edges, 2)

    # Collect perp edge pairs of the top face
    perpendicular_edge_pairs = []
    for edge_pair in potential_edge_pairs:
        if np.isclose(np.dot(get_vector_from_point_pair_idxs(box_points, edge_pair[0]),
                             get_vector_from_point_pair_idxs(box_points, edge_pair[1])), 0):
            perpendicular_edge_pairs.append(edge_pair)

    assert perpendicular_edge_pairs != False

    unique_edges = set()
    for edge_pair in perpendicular_edge_pairs:
        unique_edges.add(edge_pair[0])
        unique_edges.add(edge_pair[1])

    # assert len(unique_edges) == 4

    edge1_1, edge2_1 = perpendicular_edge_pairs.pop()

    assert np.linalg.norm(get_vector_from_point_pair_idxs(box_points, edge1_1)) != np.linalg.norm(get_vector_from_point_pair_idxs(box_points, edge2_1))

    # Find shortest edge
    if np.linalg.norm(get_vector_from_point_pair_idxs(box_points, edge1_1)) < np.linalg.norm(get_vector_from_point_pair_idxs(box_points, edge2_1)):
        top_face_shortest_edge_idx = list(edge1_1)
        top_face_longest_edge = list(edge2_1)
    else:
        top_face_shortest_edge_idx = list(edge2_1)
        top_face_longest_edge = list(edge1_1)
    top_face_shortest_edge_vec = get_vector_from_point_pair_idxs(box_points, top_face_shortest_edge_idx)
    top_face_longest_edge_vec = get_vector_from_point_pair_idxs(box_points, top_face_longest_edge)

    # Get 2 pairs of parallel edges
    parallel_edge_idxs1 = (sorted(edge1_1), sorted(get_pair_contains_pairitem(perpendicular_edge_pairs, edge2_1)))
    parallel_edge_idxs2 = (sorted(edge2_1), sorted(get_pair_contains_pairitem(perpendicular_edge_pairs, edge1_1)))

    longest_edge_frame_vecs = get_frame_from_edge(longest_edge_points_idxs_tuple,
                                                                             box_points)

    return dict(top_face_shortest_edge_idx=top_face_shortest_edge_idx,
                top_face_longest_edge=top_face_longest_edge_vec,
                parallel_edges=[parallel_edge_idxs1, parallel_edge_idxs2],
                longest_edge_as_pointpair=longest_edge_points_idxs_tuple[0],
                longest_edge_as_vec=longest_edge_points_idxs_tuple[0][0]-longest_edge_points_idxs_tuple[0][1],
                longest_edge_frame_vecs=longest_edge_frame_vecs)


def get_bbox_colored_lineset(box_points, parallel_edges_as_tuples, top_face_shortest_edge=None):
    import open3d
    parallel_edge_idxs1, parallel_edge_idxs2 = parallel_edges_as_tuples

    import sys
    if sys.version_info[0] == 2:
        allfaces_colored_lineset = map(sorted, BBOX_EDGES_AS_POINT_IDX_PAIRS)
    else:
        allfaces_colored_lineset = list(map(sorted, BBOX_EDGES_AS_POINT_IDX_PAIRS))
    try:
        # Set bbox line colors
        for edge in parallel_edge_idxs1:
            bbox_line_colors[allfaces_colored_lineset.index(edge)] = RED

        for edge in parallel_edge_idxs2:
            bbox_line_colors[allfaces_colored_lineset.index(edge)] = GREEN
        if top_face_shortest_edge is not None:
            bbox_line_colors[allfaces_colored_lineset.index(top_face_shortest_edge)] = YELLOW
    except ValueError as e:
        bandu_logger.debug(e)
        import pdb
        pdb.set_trace()


    allfaces_colored_lineset = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(box_points),
        lines=open3d.utility.Vector2iVector(allfaces_colored_lineset),
    )
    # bandu_logger.debug("BBOX LINE COLORS")
    # bandu_logger.debug(bbox_line_colors)
    allfaces_colored_lineset.colors = open3d.utility.Vector3dVector(bbox_line_colors)
    return allfaces_colored_lineset


def get_lineset(obb):
    box_points = np.asarray(obb.get_box_points())
    edge_dic = get_edge_dict(box_points)
    bbox_colored_lineset=get_bbox_colored_lineset(np.asarray(obb.get_box_points()),
                                                  edge_dic['parallel_edges'],
                                                  top_face_shortest_edge=edge_dic['top_face_shortest_edge_idx'])
    return bbox_colored_lineset, edge_dic


def get_obb_from_points(cluster_pts,
                        table_normal=None):
    import open3d
    obb = open3d.geometry.OrientedBoundingBox()

    obb = obb.create_from_points(open3d.utility.Vector3dVector(cluster_pts))

    # bandu_logger.debug("Obb R before")
    # bandu_logger.debug(obb.R)

    # if table_normal is not None:
    #     import pdb
    #     pdb.set_trace()
    #     zobb = np.dot(obb.R, np.asarray([0, 0, 1]))
    #     rot = get_rotation_matrix_between(table_normal, start_vec=zobb)
    #     obb = obb.rotate(rot)

    # bandu_logger.debug("Obb R after")
    # bandu_logger.debug(obb.R)


    # bandu_logger.debug("OBB center")
    # bandu_logger.debug(obb.get_center())
    return obb
    # ret_dict = dict(
    #     obb=obb,
    # )
    # ret_dict.update(dic)
    # return ret_dict


def get_obbs(points, labels,
             core_samples_mask,
             vis,
             add_geom=False,
             colors=None,
             plane_model=None,
             table_aligned_obb=True):
    obbs = []
    obb_colors = []
    obb_short_edges = []
    bbox_geom_list = []

    ret_list = [] # A tuple for each obb

    unique_object_labels = set(labels)
    for object_label in unique_object_labels:
        class_member_mask = (labels == object_label)
        cluster_pts = points[class_member_mask & core_samples_mask]
        cluster_colors = colors[class_member_mask & core_samples_mask]

        # Filtering logic
        min_height = -.24
        if object_label == -1 or len(cluster_pts) < 400 or \
                np.amax(cluster_pts[:, 2], axis=0) < min_height or \
                np.all(cluster_colors.mean(axis=0) < np.asarray([0.3, 0.3, 0.3])):
            continue

        # bandu_logger.debug("Cluster" + str(label)+ "colors mean" + str(cluster_colors.mean(canonical_axis=0)))

        if table_aligned_obb:
            dic = get_table_aligned_obb(cluster_pts,
                                        plane_model,
                                        vis)
        else:
            dic = get_obb_from_points(cluster_pts)
        ret_list.append(dic)
        # obb, shortest_edge_vec, longest_edge_vec, bbox_lines = lst

        # obbs.append(obb)
        # obb_colors.append(np.median(cluster_colors, canonical_axis=0))
        # obb_short_edges.append(shortest_edge_vec/np.linalg.norm(shortest_edge_vec))
        # bbox_geom_list.append(bbox_lines)

        # if add_geom:
        #     vis.add_geometry(bbox_lines)

        # if output_type(colors) is not None:
        # else:
        #     obb_colors = None
    # return obbs, obb_colors, obb_short_edges, bbox_geom_list
    return ret_list


# Raycasting Code:
from collections import namedtuple
import sys

Pt = namedtuple('Pt', 'x, y')  # Point
Edge = namedtuple('Edge', 'a, b')  # Polygon edge from a to b
Poly = namedtuple('Poly', 'name, edges')  # Polygon

_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min


def rayintersectseg(p, edge):
    ''' takes a point p=Pt() and an edge of two endpoints a,b=Pt() of a line segment returns boolean
    '''
    a, b = edge
    if a.y > b.y:
        a, b = b, a
    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    intersect = False

    if (p.y > b.y or p.y < a.y) or (
            p.x > max(a.x, b.x)):
        return False

    if p.x < min(a.x, b.x):
        intersect = True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        intersect = m_blue >= m_red
    return intersect


def _odd(x): return x % 2 == 1


def is_point_inside(p, poly):
    ln = len(poly)
    return _odd(sum(rayintersectseg(p, edge)
                    for edge in poly.edges))


def get_2d_obb(points, vis, add_geom=False):
    import open3d
    """

    Args:
        points: N x 3 points
        vis:
        add_geom:

    Returns:
        corner_points: 4 rectangle corner points
        center:

    This method projects to the eigenvectors to find the extents and center,
    then projects back to get the correctly oriented corner points in the world frame.

    """

    points = points[:, :2]
    # points: (n, 2)
    # Returns transformation matrix to new OBB
    from sklearn.decomposition.pca import PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    x_ax, y_ax = pca.components_


    eigenmatrix = np.asarray([x_ax, y_ax]) # 2x2, row vectors
    projected_points = (np.dot(eigenmatrix, points.T)).T

    maxX = np.max(projected_points[:, 0])
    maxY = np.max(projected_points[:, 1])
    minX = np.min(projected_points[:, 0])
    minY = np.min(projected_points[:, 1])

    p_topleft = [minX, maxY]
    p_topright = [maxX, maxY]
    p_botleft = [minX, minY]
    p_botright = [maxX, minY]

    # Undo projection to get the corner points in the original world frame
    corner_points = np.asarray([np.dot(eigenmatrix.T, p_topleft),
                                np.dot(eigenmatrix.T, p_topright),
                                np.dot(eigenmatrix.T, p_botleft),
                                np.dot(eigenmatrix.T, p_botright)])


    if add_geom:
        corner_points_3d = np.zeros((corner_points.shape[0], corner_points.shape[1] + 1))
        corner_points_3d[:, :2] = corner_points
        corner_points_3d[:, 2] = np.zeros_like(corner_points[:, 0])

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(corner_points_3d),
            lines=open3d.utility.Vector2iVector(EDGES_2D),
        )
        vis.add_geometry(line_set)

    center = np.dot(eigenmatrix.T, np.asarray([(maxX+minX)/2, (maxY+minY)/2]))
    extentX, extentY = (maxX - minX), (maxY - minY)
    return corner_points, center, extentX, extentY, eigenmatrix


"""
Visualization
"""

import open3d
def create_arrows_from_np(vecs, positions, color):
    out_geoms = []
    for i in range(len(vecs)):
        if isinstance(color[0], list):
            out_geoms.append(create_arrow(vecs[i], color[i], position=positions[i]))
        else:
            out_geoms.append(create_arrow(vecs[i], color, position=positions[i]))
    return out_geoms


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

    print("ln554")
    print(vec)
    rot_mat = get_rotation_matrix_between_vecs(vec, [0, 0, 1])
    print(rot_mat)
    mesh_arrow.rotate(rot_mat, center=np.array([0,0,0]))

    H = np.eye(4)
    H[:3, 3] = position
    mesh_arrow.transform(H)

    mesh_arrow.paint_uniform_color(color)
    return mesh_arrow


def create_sphere(vec, color, radius=.1):
    import open3d
    vec_len = np.linalg.norm(vec)
    sphere = open3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
    )

    H = np.eye(4)
    H[:3, 3] = vec
    sphere.transform(H)
    sphere.paint_uniform_color(color)

    return sphere
    # vis.add_geometry(sphere)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



"""
Data augmentation transforms
"""

import numpy as np


def load_scene(load_vr_bool=False,
               load_vr_gripper=False,
               realtime=1,
               p_connect_type=None):
    """
    Loads:
    - Plane
    - Table
    - Free-floating PR2 gripper
    - Bandu objects, default 2
        - Object 1
        - Object 2
    :param load_vr_bool:
    :param urdf_paths:
    :return:
    """
    assert p_connect_type is not None, p_connect_type
    if p_connect_type is not None:
        connection_id = p.connect(p_connect_type)

        if (connection_id < 0 and p_connect_type == p.SHARED_MEMORY):
            print("ln624 p.SHARED_MEMORY connect failed")
            raise NotImplementedError
            # if we failed to connect to the VR/physics sim, we must be debugging, and want to see the GUI
            p.connect(p.GUI)

    p.setAdditionalSearchPath(str(BANDU_ROOT))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    #disable rendering during loading makes it much faster
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_VR_RENDER_CONTROLLERS, 0)

    # Load URDFs
    # import pdb
    # pdb.set_trace()
    plane_id = p.loadURDF(str(BANDU_ROOT / "parts/non_bandu/plane_urdf/plane.urdf"),
                          [0.000000, 0.000000, 0.000000],
                          [0.000000, 0.000000, 0.000000, 1.000000])

    table_id = p.loadURDF(str(BANDU_ROOT / "parts/non_bandu/table/table.urdf"),
                          [0., 0, 0.000000],
                          [0.000000, 0.000000, 0.707107, 0.707107])

    bandu_logger.debug("Table ID")
    bandu_logger.debug(table_id)

    # Set table joint position (Why?)
    jointPositions = [0.000000]
    for jointIndex in range(p.getNumJoints(table_id)):
        p.resetJointState(table_id, jointIndex, jointPositions[jointIndex])


    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.setGravity(0, 0, -10)

    p.setRealTimeSimulation(realtime)

    # urdf_paths, urdf_ids = load_bandu_objects(urdf_paths=urdf_paths, num_objects=num_objects)

    # if load_vr_gripper:
    #     pr2_gripper, pr2_cid, pr2_cid2 = load_pr2_gripper()
    #
    # if load_vr_bool:
    #     assert load_vr_gripper, load_vr_gripper
        # detect_vr_controller_and_load_plugin(pr2_cid, pr2_cid2, pr2_gripper)
        # detect_vr_controller_and_sync_pb_gripper(pr2_cid, pr2_cid2, pr2_gripper)


    if load_vr_gripper and load_vr_bool:
        return plane_id, table_id, gripper_cid, pr2_gripper
    else:
        return plane_id, table_id
    # return urdf_paths, urdf_ids


def generate_tmp_urdf(urdf_file, color):
    """
    Load URDF, change properties, and return a tmp file to the new URDF
    :param urdf_file:
    :param color:
    :return:
    """
    urdf_obj = urdfpy.URDF.load(urdf_file)
    for link in urdf_obj.links:
        link.visuals[0].material.color = color

    # with tempfile.NamedTemporaryFile(mode='wt', dir_containing_pkls=F"/tmp", delete=False, suffix=".urdf") as fp:
    #     urdf_obj.save(fp)
    #     tmp_file_name = fp.name
    # return os.path.join("/tmp", tmp_file_name)
    tmp_file_path = f"/tmp/{os.path.basename(urdf_file)}_{time.strftime('%Y%m%d-%H%M%S')}_{np.random.randint(1E5)}.urdf"
    urdf_obj.save(tmp_file_path)
    return tmp_file_path

import trimesh


def get_max_mesh_points(stl_paths):
    max_num_points = 0
    mesh_list = load_meshes_vertices(stl_paths)

    for mesh in mesh_list:
        max_num_points = max(mesh.shape[0], max_num_points)
    return max_num_points


def get_mesh_num_points_list(stl_paths):
    mesh_list = load_meshes_vertices(stl_paths)
    mesh_num_points_list = []
    for mesh in mesh_list:
        mesh_num_points_list.append(mesh.shape[0])
    return mesh_num_points_list


def load_meshes_vertices(stl_paths, randomize_mesh_indices=True):
    try:
        stl_paths = [stl_path for stl_path in stl_paths]
        for path in stl_paths:
            assert os.path.isfile(path)
    except:
        print("ln719 excepting")
        stl_paths = [str(BANDU_ROOT / stl_path) for stl_path in stl_paths]

    mesh_vertices_centered = []
    for stl_path in stl_paths:
        with open(stl_path, "rb") as fp:
            mesh = trimesh.load_mesh(fp, file_type="STL")
        mvc = np.array(mesh.vertices - mesh.center_mass)

        if randomize_mesh_indices:
            np.random.shuffle(mvc)
        mesh_vertices_centered.append(mvc)
    return mesh_vertices_centered


def load_centered_meshes(stl_paths):
    try:
        stl_paths = [stl_path for stl_path in stl_paths]
        for path in stl_paths:
           assert os.path.isfile(path)
    except:
        stl_paths = [str(BANDU_ROOT / stl_path) for stl_path in stl_paths]

    meshes_centered = []
    for stl_path in stl_paths:
        with open(stl_path, "rb") as fp:
            mesh = trimesh.load_mesh(fp, file_type="STL")
        mesh.apply_translation(-mesh.center_mass)

        meshes_centered.append(mesh)
    return meshes_centered


def load_bandu_objects(table_offset=(-.8, .2, 0),
                       object_offset=(0, .4, 0),
                       urdf_paths=None,
                       colors=None,
                       scale_factor=1.0,
                       read_extra_config=False,
                       mass_coeff=.1):
    """

    :param table_offset:
    :param object_offset:
    :return: urdf_paths (list of Path objects), urdf_Ids (list of integers). Optional argument to load specific urdfs instead of randomly sampling
    """
    p.setAdditionalSearchPath(str(BANDU_ROOT))

    assert (isinstance(urdf_paths, tuple) or isinstance(urdf_paths, list)), bandu_logger.debug(f"{urdf_paths} \n Are you passing in a single string? Wrap it in an iterable.")
    if colors:
        assert len(colors) == len(urdf_paths), (len(colors), len(urdf_paths))
    # Import bandu meshes
    # urdf_dir_ = BANDU_ROOT / "parts/urdfs/**/*.urdf"

    num_objects = len(urdf_paths)
    # If currently Linux (posix), convert windows paths to Linux. Assumes vr files always generated on windows
    if os.name == "posix":
        urdf_paths = [convert_windowspath_to_posixpath(_) for _ in urdf_paths]
    # urdf_paths.reverse()
    pybullet_body_ids_list = []

    # store extra config per body
    extra_config_list = []

    for obj_idx in range(num_objects):
        # bandu_logger.debug("Attempting to load..." + str(urdf_paths[obj_idx]))
        bandu_logger.debug("Attempting to load..." + str(urdf_paths[obj_idx]))
        pos = [1.300000 + table_offset[0] + object_offset[0] - .2 * (obj_idx % 4),
               -0.700000 + table_offset[1] + object_offset[1] + .2 * math.floor(obj_idx / 4),
               0.750000]
        quat = [0.000000, 0.707107, 0.000000, 0.707107]

        try:
            if colors:
                # Create a temporary URDF with the adjusted color
                tmp_urdf_path = generate_tmp_urdf(urdf_paths[obj_idx], colors[obj_idx])
                urdf_id = p.loadURDF(tmp_urdf_path, pos, quat, globalScaling=scale_factor)
                os.remove(tmp_urdf_path)
            else:
                bandu_logger.debug(urdf_paths[obj_idx])
                # print(f"ln786 {urdf_paths[obj_idx]}")
                # we cannot have backward slashes
                urdf_id = p.loadURDF(urdf_paths[obj_idx], pos, quat, globalScaling=scale_factor)
        except p.error as e:
            bandu_logger.debug(f"{urdf_paths[obj_idx]} failed to load.")
            raise Exception

        if p.getNumJoints(urdf_id) > 1:
            print("ln795")
            print(urdf_paths[obj_idx])
        mass = p.getDynamicsInfo(urdf_id, -1)[0]
        new_mass = mass_coeff * mass
        p.changeDynamics(urdf_id, -1, new_mass)

        bandu_logger.debug("Loaded ID " + str(urdf_id))
        pybullet_body_ids_list.append(urdf_id)
        time.sleep(.1)

        if read_extra_config:
            dir_ = os.path.dirname(urdf_paths[obj_idx])
            config_path = Path(dir_) / "extra_config"

            # if the below logic fails, that's because we didn't label the objects yet
            # with their extra config
            with open(str(config_path), "r") as fp:
                jd = json.load(fp)
                extra_config_list.append(jd)
    # return [Path(PureWindowsPath(_)) for _ in urdf_paths], urdf_ids
    assert pybullet_body_ids_list == sorted(pybullet_body_ids_list), f"{str(pybullet_body_ids_list, sorted(pybullet_body_ids_list))} If URDF IDs aren't sorted in increasing order, you removed bodies incorrectly"
    # return urdf_paths, urdf_ids

    bandu_logger.debug(f"Num bodies: {p.getNumBodies()}")
    if read_extra_config:
        return pybullet_body_ids_list, extra_config_list
    else:
        return pybullet_body_ids_list


def parse_urdf_xml_for_object_name(urdf_xml_path):
    from lxml import etree

    if urdf_xml_path[0] == "/":
        tree = etree.parse(urdf_xml_path)
    else:
        try:
            tree = etree.parse(urdf_xml_path)
        except:
            tree = etree.parse(str(BANDU_ROOT) + "/" + urdf_xml_path)
    return tree.find("name").text


def get_object_names(urdf_paths_as_tuple):
    # returns object names, in same order as input tuple
    object_names = []
    for urdf_path in urdf_paths_as_tuple:
        object_name = parse_urdf_xml_for_object_name(os.path.dirname(urdf_path) + "/model.config")
        object_names.append(object_name)
    return object_names


def remove_bandu_objects(num_bandu_objects_to_remove, total_bandu_objects=None, total_scene_objects = 2):
    """
    Remove bandu objects by
    :param num_bandu_objects_to_remove:
    :return:
    """
    # non_bandu_objects = 2
    if total_bandu_objects is None:
        total_bandu_objects = num_bandu_objects_to_remove

    body_ids = list(range(total_bandu_objects))
    body_ids = [total_scene_objects + x for x in body_ids]
    body_ids.reverse()
    # import pdb
    # pdb.set_trace()
    # for _ in body_ids:
    #     p.removeBody(_)
    for i in range(num_bandu_objects_to_remove):
        p.removeBody(body_ids[i])


def normalize_along_rows(mat):
    assert len(mat.shape) == 2
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


from bandu.config import TABLE_HEIGHT

def get_number_stacked_objects(table_id, plane_id, total_objs):
    """
    :param table_id:
    :param plane_id:
    :param total_objs: Total number of Bandu objects
    :return:
    """

    # counts the number of objects which are NOT in contact with the table
    # adds number of base objects allowed to touch the table
    # to get final number of stacked objects
    cp = p.getContactPoints()

    bodies_found_on_table = [tup[2] for tup in cp if (tup[1] == table_id and tup[1] != plane_id)]
    out = total_objs - len(set(bodies_found_on_table)) + 1
    assert out > 0
    print("ln874 num stacked")
    print(out)
    return out

def get_tower_height(object_ids, keep_table_height=False, scale=True):
    print("ln899 this isn't right")
    raise NotImplementedError
    # gets top height of all objects in the scene
    # in cm
    # if not object_ids.numel():
    if not object_ids:
        return TABLE_HEIGHT
    overall_top = -float("inf")

    for obj_id in object_ids:
        # aabb = p.getAABB(obj_id)
        # aabbMinVec = aabb[0]
        # aabbMaxVec = aabb[1]
        # object_height = aabbMaxVec[-1]

        overall_top = max(overall_top, object_height)

    candidate_height = overall_top
    if not keep_table_height:
        candidate_height = candidate_height - TABLE_HEIGHT
    if scale:
        candidate_height = candidate_height * 100
    return candidate_height
    # return (overall_top-TABLE_HEIGHT)*100

import torch
def unit_radius(centered_vertices):
    # Both centered and radius 1
    # Take max over all the centered vertex lengths
    radii = torch.norm(centered_vertices, dim=-1)

    max_radius = torch.max(radii, -1)[0]

    # -> [batch_size, num_objects, num_points, 1]
    max_radius_vectorized = max_radius.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, centered_vertices.shape[-2], -1)

    # Divide each vector by the max radius
    centered_vertices = centered_vertices/max_radius_vectorized

    new_radii = torch.norm(centered_vertices, dim=-1)
    new_max_radius = torch.max(new_radii, -1)[0]
    assert torch.all(new_max_radius <= 1 + 1E-3), (new_max_radius, max_radius_vectorized)
    return centered_vertices


from scipy.spatial.transform import Rotation as R
def get_initial_orientation(start_rotation_type, **kwargs):
    if start_rotation_type == "full":
        initial_orientation = R.random()
    elif start_rotation_type == "planar":
        initial_orientation = R.from_quat(random_z_axis_orientation(2 * np.pi)[0])
    elif start_rotation_type == "identity":
        initial_orientation = R.from_euler("xyz", [0,0,0])
    elif start_rotation_type == "flat_or_up":
        initial_orientation = random.choice([R.from_euler("xyz", [0 ,0 ,0]), R.from_euler("xyz", [np.pi/2, 0, 0])])
    elif start_rotation_type == "close_to_target":
        initial_orientation = R.from_rotvec(misc_util.random_three_vector() * np.random.uniform(low=0.0, high=(2*np.pi/10))) \
                              * R.from_quat(kwargs['target_quat_original_order'][kwargs['obj_idx']])
    else:
        raise NotImplementedError
    return initial_orientation


def get_positive_indices(normals, mesh_triangles_center, object_name):
    # This should ONLY be applied to the canonical mesh normals
    labels = []
    if object_name == "Skewed Rectangular Prism":
        for idx, normal in enumerate(normals):
            if np.all(np.isclose(normal, [0, 0, 1])) or np.all(np.isclose(normal, [0, 0, -1])):
                labels.append(idx)
    elif object_name == "Bandu Block":
        for idx, normal in enumerate(normals):
            if np.all(np.isclose(normal, [0, 1, 0])) or np.all(np.isclose(normal, [0, -1, 0])):
                labels.append(idx)
    elif object_name == "Colored_Block":
        for idx, normal in enumerate(normals):
            if (np.all(np.isclose(normal, [0, 1, 0])) or np.all(np.isclose(normal, [0, -1, 0]))):
                labels.append(idx)
    elif object_name == "Pencil":
        # np.concatenate((normals, mesh_triangles_center), axis=-1)
        for idx, normal in enumerate(normals):
            # if np.sum(normal == 0) == 2:
                # print("ln 912 normal")
                # print(normal)
            if np.all(np.isclose(normal, [0, 1, 0])) and mesh_triangles_center[idx][1] > 0:
                labels.append(idx)
    elif object_name == "Skewed Cylinder":
        for idx, normal in enumerate(normals):
            if np.all(np.isclose(normal, [0, 0, -1])) or np.all(np.isclose(normal, [0, 0, 1])):
                labels.append(idx)
        assert labels, labels
    elif object_name == "Skewed Triangular Prism":
        for idx, normal in enumerate(normals):
            if np.all(np.isclose(normal, [0, 0, 1])):
                labels.append(idx)
    elif object_name == "Skewed Wedge":
        for idx, normal in enumerate(normals):
            if np.all(np.isclose(normal, [1, 0, 0])) or np.all(np.isclose(normal, [-1, 0, 0])):
                labels.append(idx)
    else:
        # print(f"ln930 {object_name}")
        raise NotImplementedError
    # print(f"ln932 {object_name} surface normal index {labels}")
    # print(f"found normal {normals[labels]}")
    assert labels, labels
    return labels