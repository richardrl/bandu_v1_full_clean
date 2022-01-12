import gym
import pymesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from supervised_training.mesh_gen.generate_meshes import view_mesh
from mayavi import mlab
import trimesh
import numpy as np
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from supervised_training.utils import mesh_util
import math
from bandu.utils import pb_util

def get_min_max_corners_from_lwh(l, w, h):
    assert l > 0 and w > 0 and h > 0
    min_p = np.zeros(3)
    max_p = np.zeros(3)

    min_p[0] = -l/2
    min_p[1] = -w/2
    min_p[2] = -h/2

    max_p[0] = l/2
    max_p[1] = w/2
    max_p[2] = h/2
    return min_p, max_p


def get_lwh_from_min_max_corners(max_corner):
    l = 2*max_corner[0]
    w = 2*max_corner[1]
    h = 2*max_corner[2]
    return l, w, h


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

# n_points = 10000
# n_dim = 10
# Z = np.random.rand(n_points,n_dim)
# x = np.random.rand(n_dim)
# print(in_hull(Z, x))

import copy
import os

class MeshMakerEnv(gym.Env):
    def __init__(self, list_of_mesh_paths, trained_network):
        """

        :param main_mesh: Path to STL file
        """
        # self.main_mesh = pymesh.load_mesh(main_mesh)
        self.list_of_mesh_paths = list_of_mesh_paths
        self.trained_network = trained_network

    def reset(self):
        # self.active_mesh = pymesh.load_mesh(np.random.choice(self.list_of_mesh_paths))
        self.active_mesh = trimesh.load_mesh(np.random.choice(self.list_of_mesh_paths))
        self.mesh_pieces = [trimesh.Trimesh(vertices=self.active_mesh.vertices, faces=self.active_mesh.faces)]
        self.mesh_pos = [np.zeros(3)]
        self.mesh_quat = [np.array([0., 0., 0., 1.])]
        return self.active_mesh

    def serialize_mesh_pieces(self, save_dir):
        for pidx, piece in enumerate(self.mesh_pieces):
            pymesh.meshio.save_mesh(os.path.join(save_dir, f"{pidx}.stl"), piece)

    def step(self, action: dict):
        """

        :param action: Dictionary of:
            lwh: list of lwh tuples for each object to be added, corresponding to x, y, z dimensions
            pos: list of XYZ tuples
            quat: list of 4D quaternion
        :return:
        """
        lwh_tuples = action['lwh']
        pos_tuples = action['pos']
        # quat_tuples = action['quat']
        rotvec_tuples = action['rotvec']

        num_objects = len(lwh_tuples)

        assert num_objects > 0

        # whether or not the added mesh touches
        null_intersection_exists = False

        for idx in range(num_objects):
            l, w, h = lwh_tuples[idx]
            pos = pos_tuples[idx]
            # quat = quat_tuples[idx]
            rotvec = rotvec_tuples[idx]

            # bm = pymesh.generate_box_mesh(*get_min_max_corners_from_lwh(l, w, h))

            T = np.zeros((4,4))
            T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
            T[:3, 3] = pos

            bm = trimesh.creation.box(extents=[l, w, h], transform=T)

            # transform
            # new_vertices = R.from_quat(quat).apply(bm.vertices) + pos
            # new_vertices = R.from_rotvec(rotvec).apply(bm.vertices - bm.centroid) + pos

            # new_mesh = pymesh.form_mesh(new_vertices, bm.faces)

            # intersection = pymesh.boolean(self.active_mesh, bm, "intersection")

            intersection = self.active_mesh.intersection(bm)
            # cm = trimesh.collision.CollisionManager()
            #
            # cm.add_object(bm)
            #
            # cm.add_object(self.active_mesh)

            # intersection = cm.in_collision_internal()

            # print("ln100 intersection")
            # print(intersection.vertices)

            print("ln132 int type")
            print(type(intersection))
            try:
                if isinstance(intersection, trimesh.Trimesh):
                    if len(intersection.vertices) == 0:
                        null_intersection_exists = True
                elif isinstance(intersection, trimesh.Scene):
                    if intersection.is_empty:
                        null_intersection_exists = True
            except:
                import pdb
                pdb.set_trace()

            self.mesh_pieces.append(bm)
            self.mesh_pos.append(pos)
            self.mesh_quat.append(R.from_rotvec(rotvec).as_quat())

            # repeatedly concatenate the mesh
            # self.active_mesh = pymesh.merge_meshes([self.active_mesh, new_mesh])
            # self.active_mesh = trimesh.boolean.union([self.active_mesh, bm])
            self.active_mesh = trimesh.util.concatenate([self.active_mesh, bm])

        # subdivide
        # prev_faces = self.active_mesh.faces.shape[0]
        # target_faces = 10000
        # subdivided_mesh = pymesh.subdivide(self.active_mesh, order=math.ceil(target_faces/prev_faces))
        # subdivided_mesh = trimesh.remesh.subdivide(self.active_mesh, order=math.ceil(target_faces/prev_faces))

        subdivided_mesh = self.active_mesh
        while subdivided_mesh.vertices.shape[0] < 10000:
            new_verts, new_faces = trimesh.remesh.subdivide(subdivided_mesh.vertices, subdivided_mesh.faces)
            subdivided_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)

        # check if stable using pyBullet
        # forward simulate it 500 timesteps OR
        # what about stable equilibrium...?
        # static stability: ensuring the support polygon is above a certain size


        # project contact points downwards onto the XY plane
        # check curvature

        # tm = trimesh.Trimesh(vertices=self.active_mesh.vertices, faces=self.active_mesh.faces)

        # compute center of mass
        com = self.active_mesh.center_mass
        # project COM down to XY plane
        com_xy = com[:2]

        # compute contact points by slicing
        min_z = np.min(self.active_mesh.vertices[:, 2])

        max_z = np.max(self.active_mesh.vertices[:, 2])

        height = max_z - min_z

        # contact_point_threshold = (height * 1/10) + min_z
        contact_point_threshold = .005 + min_z

        pc = trimesh.sample.sample_surface(self.active_mesh, 1000)[0]

        # TODO: average Gaussian curvature of contact surface should be 0

        # TODO: maybe calculate using dense pointcloud here is better
        contact_points_pc = pc[pc[:, 2] < contact_point_threshold]

        # calculate curvature off subdivided mesh
        # subdivided_mesh.add_attribute("vertex_mean_curvature")
        #
        circle_rad = .02

        cond = np.logical_and(subdivided_mesh.vertices[:, 2] < contact_point_threshold,
                              np.linalg.norm(subdivided_mesh.vertices[:, :2] - com_xy, axis=-1) < circle_rad)

        if np.sum(cond) == 0:
            print("Didn't find any contact points close to COM")

            s = dict(active_mesh=pymesh.form_mesh(self.active_mesh.vertices, self.active_mesh.faces),
                     mesh_pieces=copy.deepcopy(self.mesh_pieces),
                     mesh_pos=copy.deepcopy(self.mesh_pos),
                     mesh_quat=copy.deepcopy(self.mesh_quat)
                     )
            return s, 0, False, dict()


        curvature_values_of_points_in_circle_below_com = trimesh.curvature.discrete_mean_curvature_measure(subdivided_mesh, subdivided_mesh.vertices[cond], .001)

        # contact_points_mesh_curvature_circle = subdivided_mesh.get_attribute("vertex_mean_curvature")[cond]
        # end curvature calculation

        contact_points_xy = contact_points_pc[:, :2]

        # print("ln156 contact_points_xy")
        # print(contact_points_xy)
        # print(com_xy)
        stable = in_hull(contact_points_xy, com_xy)

        # check if height maximizing along stable axis
        # use PCA + find eigenvector
        # check if the cosine distance is beyond a threshold
        # you can't calculate HM without calculating all the possible stable faces...
        # we could use goldberg to calculate them...

        mesh_copy = trimesh.Trimesh(self.active_mesh.vertices, self.active_mesh.faces)
        transformed_mesh, tm_height = mesh_util.get_tallest_stable_transformed_mesh(mesh_copy)
        if float('%.3g' % height) >= float('%.3g' % tm_height):
            tallest_height = 1
        else:
            tallest_height = 0

        # also, disable picking an object where the contact surface is curved

        if np.mean(curvature_values_of_points_in_circle_below_com) < 1:
            flat_bottom = 1
        else:
            flat_bottom = 0

        # pca = PCA(n_components=3)
        # pca.fit(pc)
        # hm = False

        # reward = stable * hm * null_intersection_exists
        print("ln150 stable")
        print(stable)

        # the only way to get the height of the longest stable face, is to calculate all stable faces..
        # use goldberg to calculate

        print("ln153 nie")
        print(null_intersection_exists)

        print("ln184 tallest height")
        print(height)
        print(tm_height)
        print(tallest_height)

        print("ln187 flat bottom")
        print(flat_bottom)
        # print(contact_points_mesh_curvature_circle)
        # print(self.active_mesh.get_attribute("vertex_mean_curvature"))

        reward = stable * (not null_intersection_exists) * tallest_height * flat_bottom

        print("ln146 Reward")
        print(reward)
        done = True if reward == 1 else False
        info = dict(pc=pc,
                    contact_points=contact_points_pc,
                    com=com)

        s = dict(active_mesh=pymesh.form_mesh(self.active_mesh.vertices, self.active_mesh.faces),
                 mesh_pieces=copy.deepcopy(self.mesh_pieces),
                 mesh_pos=copy.deepcopy(self.mesh_pos),
                 mesh_quat=copy.deepcopy(self.mesh_quat)
                 )
        return s, reward, done, info
        # view_mesh(self.main_mesh)