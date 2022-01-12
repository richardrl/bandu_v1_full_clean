from bandu.config import *
import os
import numpy as np
import pybullet as p
import pybullet_data
import math
import urdfpy
import time
import json
import trimesh


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
    # if os.name == "posix":
    #     urdf_paths = [convert_windowspath_to_posixpath(_) for _ in urdf_paths]
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


def get_object_names(urdf_paths_as_tuple):
    # returns object names, in same order as input tuple
    object_names = []
    for urdf_path in urdf_paths_as_tuple:
        object_name = parse_urdf_xml_for_object_name(os.path.dirname(urdf_path) + "/model.config")
        object_names.append(object_name)
    return object_names

import random
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


def gen_2d_xy(x_min=TABLE_X_MIN, x_max=TABLE_X_MAX, y_min=TABLE_Y_MIN, y_max=TABLE_Y_MAX):
    """
    Randomize object positions such that COM is on the table.
    :return: Numpy array
    """
    i = 0
    X_sampled = np.random.uniform(x_min, x_max)
    Y_sampled = np.random.uniform(y_min, y_max)
    start_pos = np.zeros(2)
    start_pos[0] = X_sampled
    start_pos[1] = Y_sampled

    print("ln81 start_pos")
    print(start_pos)
    return start_pos


def get_position_for_drop_above_table(height_off_set=.05, avoid_center_amount=.3):
    ret_arr = np.zeros(3)

    if avoid_center_amount:
        valid_xy = gen_2d_xy()
        while np.linalg.norm(valid_xy - np.zeros(2)) < avoid_center_amount:
            valid_xy = gen_2d_xy()
        ret_arr[:2] = valid_xy
    else:
        ret_arr[:2] = gen_2d_xy()
    ret_arr[2] = TABLE_HEIGHT + height_off_set
    return ret_arr

def get_position_for_drop_above_table_avoid_obj(avoid_obj, height_off_set=.05):
    ret_arr = np.zeros(3)

    valid_xy = gen_2d_xy()
    ret_arr[2] = TABLE_HEIGHT + height_off_set

    while bool(p.getContactPoints(avoid_obj, pb_oid)):
        valid_xy = gen_2d_xy()

    ret_arr[:2] = valid_xy

    return ret_arr


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


def get_position_for_drop_above_table(height_off_set=.05, avoid_center_amount=.3):
    ret_arr = np.zeros(3)

    if avoid_center_amount:
        valid_xy = gen_2d_xy()
        while np.linalg.norm(valid_xy - np.zeros(2)) < avoid_center_amount:
            valid_xy = gen_2d_xy()
        ret_arr[:2] = valid_xy
    else:
        ret_arr[:2] = gen_2d_xy()
    ret_arr[2] = TABLE_HEIGHT + height_off_set
    return ret_arr


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
    plane_id = p.loadURDF(str(BANDU_ROOT / "parts/scene/plane_urdf/plane.urdf"),
                          [0.000000, 0.000000, 0.000000],
                          [0.000000, 0.000000, 0.000000, 1.000000])

    table_id = p.loadURDF(str(BANDU_ROOT / "parts/scene/table/table.urdf"),
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