import numpy as np


def get_tallest_stable_transformed_mesh(mesh):
    transforms, probs = mesh.compute_stable_poses()

    max_height = float("-inf")
    max_idx = None
    max_transform = None
    for idx, transform in enumerate(transforms):
        print(f"ln10 idx {idx}")
        mesh.apply_transform(transform)
        obj_height = mesh.bounds[1][-1] - mesh.bounds[0][-1]
        if obj_height > max_height:
            max_height = obj_height
            max_idx = idx
            max_transform = transform
        mesh.apply_transform(np.linalg.inv(transform))
        mesh.vertices -= mesh.center_mass

    mesh.apply_transform(transforms[max_idx])
    mesh.vertices -= mesh.center_mass
    return mesh, max_height, max_transform

import itertools

def get_stable_transforms_sorted_by_height(mesh):
    transforms, probs = mesh.compute_stable_poses()

    inverted_transforms = [np.linalg.inv(trans) for trans in transforms]

    mixed_transforms = list(itertools.chain(*zip(transforms, inverted_transforms)))

    found_heights = []
    found_transforms = []

    for idx, transform in enumerate(mixed_transforms):
        print(f"ln10 idx {idx}")
        mesh.apply_transform(transform)
        obj_height = mesh.bounds[1][-1] - mesh.bounds[0][-1]

        found_heights.append(obj_height)
        found_transforms.append(transform)

        mesh.apply_transform(np.linalg.inv(transform))
        mesh.vertices -= mesh.center_mass

    zipped = zip(found_heights, found_transforms)

    sorted_ = sorted(zipped, key=lambda tup: tup[0], reverse=True)
    return zip(*sorted_)



"""
Copied from spinningup_private
"""

import copy

import numpy
import trimesh
from trimesh.exchange.urdf import export_urdf
from trimesh.exchange.obj import export_obj
import os
from copy import deepcopy
import numpy as np
from bandu.config import BANDU_ROOT
import click

from utils.bandu_util import get_positive_indices

parts_dir_path = BANDU_ROOT / "parts"
# Convert entire directory of objects into convexified urdf folders
stl_input_dir_ = str(parts_dir_path / "stls")
from bandu.config import bandu_logger


def stable_poses_from_stl(stl_path):
    mesh = trimesh.load_mesh(stl_path)
    mesh2 = deepcopy(mesh)
    stable_out = trimesh.poses.compute_stable_poses(mesh2)

    # Take first transform
    # transform = stable_out[0][0]
    # homogenized_vertices = np.concatenate(
    #     (
    #         mesh2.vertices, np.ones((mesh2.vertices.shape[0], 1))
    #     ), canonical_axis=-1)
    # mesh2.vertices = (homogenized_vertices @ transform)[:, :3]
    return stable_out


def mesh_processing(mesh):
    mesh.vertices -= mesh.center_mass
    # mesh.density = 5000
    mesh.density = 770


@click.group()
# @click.pass_context #For some reason this line messes things up
def cli():
    # ctx.ensure_object(dict)
    pass
from pathlib import Path
# @cli.command('single_stl2output_format') # By default click changes the underscore to a hyphen for some reason..
# @click.argument("stl_full_path")
# @click.argument("output_format")
# @click.argument("filename")
# @click.argument("output_dir")
def convert_stl_file(stl_full_path, output_format, filename, output_dir, scale=1,
                     resolution=16000000):
    """

    :param stl_full_path:
    :param output_format:
    :param filename: E.g. "Colored Block.STL"
    :return:
    """
    # mesh = trimesh.load(stl_full_path, force='mesh')

    mesh = trimesh.load(stl_full_path)

    mesh.apply_scale(scale)

    mesh_processing(mesh)

    # str(parts_dir_path / (output_format + "s") / filename[:-4])
    if output_dir:
        output_file_path_as_str = Path(output_dir) / filename[:-4]
    else:
        output_file_path_as_str = str(parts_dir_path / (output_format + "s") / filename[:-4])
    if output_format == "urdf":
        # this functions runs vhacd
        vhacd_kwargs = dict(
            resolution=resolution,
            maxhulls=1000,
            convexhullDownsampling=16,
            gamma=.00001,
            concavity=0.000001,
            targetNTrianglesDecimatedMesh = 500,
            maxNumVerticesPerCH=1024,
            mode=0, # mode 1 is way slower
            oclAcceleration=1,
            # depth = 5,
            # maxConcavity =0.04,
            # invertInputFaces = False,
            # posSampling = 5,
            # angleSampling = 5,
            # posRefine = 3,
            # angleRefine = 3,
            # alpha = 0.04,
        )
        # vhacd_kwargs = dict()
        export_urdf(mesh, output_file_path_as_str, **vhacd_kwargs)
        bandu_logger.debug("Converted to urdf " + str(filename))
    elif output_format == "obj":
        obj_as_str = export_obj(mesh)
        with open(str(parts_dir_path / (output_format + "s") / (filename[:-4] + ".obj")), 'wb') as fp:
            trimesh.util.write_encoded(fp, obj_as_str)
        bandu_logger.debug("Converted to obj " + str(filename[:-4] + ".obj"))


# @cli.command('convert_stl_folder') # By default click changes the underscore to a hyphen for some reason..
# @click.argument("output_format")
# @click.argument("input_dir")
# @click.argument("output_dir")
# @click.pass_context
import pathlib

def convert_stl_folder(output_format, input_dir, output_dir, scale=1,
                       resolution=16000000):
    """
    Converts folder of STLs to URDFs. Assumes STLs are in root of input_dir
    :param output_format:
    :param input_dir: Input directory of STLS. Assumes all STLs are at the root level of this input directory.
    :return:
    """
    odp = pathlib.Path(output_dir)
    odp.mkdir(exist_ok=True)

    print(f"ln114 using scale {scale}")
    if not input_dir:
        input_dir = stl_input_dir_
    for filename in os.listdir(input_dir):
        if filename[-3:].lower() not in ["stl"]:
            bandu_logger.debug("Skipped " + str(filename))
            continue
        stl_full_path = os.path.join(input_dir, filename)


        convert_stl_file(stl_full_path, output_format, filename, output_dir, scale=float(scale),
                         resolution=resolution)
        # ctx.invoke(convert_stl_file, stl_full_path, output_format, filename, output_dir)

@cli.command('stl_info') # By default click changes the underscore to a hyphen for some reason..
def stl_info():
    # Display names of mesh and number of vertices
    for filename in os.listdir(stl_input_dir_):
        if filename[-3:].lower() not in ["stl"]:
            bandu_logger.debug("Skipped " + str(filename))
            continue
        full_path = os.path.join(stl_input_dir_, filename)
        mesh = trimesh.load_mesh(full_path)
        import pdb
        pdb.set_trace()
        mesh2 = deepcopy(mesh)

        bandu_logger.debug(filename + " num vertices: " + str(np.asarray(mesh2.vertices).shape[0]))

@cli.command('gen_sizes') # By default click changes the underscore to a hyphen for some reason..
@click.argument("stl_path")
@click.argument("scales",
                nargs=-1,
                type=float)
def gen_sizes(stl_path, scales):
    # import pdb
    # pdb.step()
    bandu_logger.debug("Running")
    """
    Generate new sizes for a mesh
    :return:
    """
    # The stl path must have r prefix if it is a string
    mesh = trimesh.load_mesh(stl_path)
    mesh_processing(mesh)

    for scale in scales:
        mesh.apply_scale(scale)
        # single_stl2output_format(stl_full_path=stl_path,
        #                          output_format="urdf",
        #                          filename=os.path.basename(stl_path)[:-4] + f"_scale{str(scale).replace('.', '_')}" + ".stl")
        filename = os.path.basename(stl_path)[:-4] + f"_scale{str(scale).replace('.', '_')}" + ".stl"
        output_format = "urdf"
        export_urdf(mesh, str(parts_dir_path / (output_format + "s") / filename[:-4]))
        bandu_logger.debug("Converted to urdf " + str(filename))
# cli.add_command(convert_stl_folder)
# cli.add_command(gen_sizes)
# cli.add_command(stl_info)


from scipy.spatial.transform.rotation import Rotation as R


class NormalStorage:
    def __init__(self, trimesh_mesh):
        self.triangles_center = deepcopy(trimesh_mesh.triangles_center)
        self.face_normals = deepcopy(trimesh_mesh.face_normals)

        # If the tricenters and trinormals are centered at the origin, we don't need to retain COM
        # self.com = com

    def rotate(self, rotation_quat):
        """
        Returns triangle centers, normal vectors
        """
        return rotate_normals(self.triangles_center, self.face_normals, rotation_quat)

    def rotate_in_place(self, rotation_quat):
        # Update new triangles_center and new_normals
        rotated_inner_ball, new_normals = self.rotate(rotation_quat)
        self.triangles_center = rotated_inner_ball
        self.face_normals = new_normals


def fix_normal_orientation(triangles_centers, normals, com):
    assert len(normals.shape) == 2
    out_normals = []

    for idx, normal in enumerate(normals):
        ray_to_com = triangles_centers[idx] - com
        # bandu_logger.debug("mesh_util 157")
        # bandu_logger.debug(np.dot(ray_to_com, normal) < 0)
        if np.dot(ray_to_com, normal) > 0:
            out_normals.append(normal)
        else:
            out_normals.append(-normal)
    return np.array(out_normals)


def rotate_normals(triangles_centers, face_normals, rotation_quat, ret_outer_ball=False):
    """
    Assumes triangles_centers are centered at world origin
    :param triangles_centers: nO x 3
    :param face_normals: nO x 3
    :param rotation_quat: nO x 4
    :return:
    """

    # Stick the normals onto the tris to get a new spiky ball
    spiky_outer_ball = triangles_centers + face_normals

    # Rotate the spiky outer ball about the COM
    rotated_spiky_outer_ball = R.from_quat(rotation_quat).apply(spiky_outer_ball)

    # Rotate the inner ball about the COM
    # the inner ball are the new triangle centers
    rotated_inner_ball = R.from_quat(rotation_quat).apply(triangles_centers)

    # Subtract inner ball from spiky ball to get the new normals
    new_normals = rotated_spiky_outer_ball - rotated_inner_ball

    new_normals = fix_normal_orientation(rotated_inner_ball, new_normals, [0,0,0.0])

    if ret_outer_ball:
        return rotated_inner_ball, new_normals, spiky_outer_ball
    else:
        return rotated_inner_ball, new_normals


if __name__ == "__main__":
    import sys
    args = sys.argv
    print("ln245 Pass everything in as args, even kwargs")
    globals()[args[1]](*args[2:])

from utils.misc_util import pad_same_size
def update_tricenters_and_normals(bandu_object_names, canonical_mesh_objects, quats_to_apply, npify=False):
    """

    :param canonical_mesh_objects:
    :param quats_to_apply:
    :param npify: for all mesh outputs, pad and convert to numpy array
    :return:
    """
    mtc_arr = [] #mesh triangle centers
    mfn_arr = [] # mesh face normals

    labels_arr = []
    transformed_meshtricenters_arr = []
    transformed_meshfacenormals_arr = []
    positive_index_arr = []

    # for obj_idx, mesh in enumerate(ret_dict['canonical_mesh_objects']):
    for obj_idx, mesh in enumerate(canonical_mesh_objects):
        num_normals = mesh.face_normals.shape[0]
        mtc = copy.deepcopy(mesh.triangles_center)
        mfn = copy.deepcopy(mesh.face_normals)
        mtc_arr.append(mtc)
        normal_labels = np.zeros(num_normals)

        normals = fix_normal_orientation(mtc, mfn, np.zeros(3))

        assert num_normals == normals.shape[0]

        mfn_arr.append(normals)

        # start_mtc, start_mfn = mesh_util.rotate_normals(mtc,
        #                                                 mfn,
        #                                                 ret_dict['start_quat'][obj_idx])
        start_mtc, start_mfn = rotate_normals(mtc, mfn, quats_to_apply[obj_idx])
        transformed_meshtricenters_arr.append(start_mtc)
        transformed_meshfacenormals_arr.append(start_mfn)

        # pick a random index out of all matched indices
        try:
            positive_idx = np.random.choice(get_positive_indices(normals, mtc, bandu_object_names[obj_idx]))

            positive_index_arr.append(positive_idx)

            normal_labels[positive_idx] = 1

            labels_arr.append(normal_labels)
        except:
            pass



    return copy.deepcopy(np.stack(pad_same_size(mtc_arr), axis=0)), \
           copy.deepcopy(np.stack(pad_same_size(mfn_arr), axis=0)), \
           copy.deepcopy(np.stack(pad_same_size(labels_arr), axis=0)) if labels_arr else None, \
           copy.deepcopy(np.stack(pad_same_size(transformed_meshtricenters_arr), axis=0)), \
           copy.deepcopy(np.stack(pad_same_size(transformed_meshfacenormals_arr), axis=0)), \
           copy.deepcopy(np.stack(positive_index_arr, axis=0)) if positive_index_arr else None