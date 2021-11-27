import numpy as np
import importlib


def pad_same_size(list_of_np_arrays):
    """

    :param list_of_np_arrays: Assumes input to be one of the following
        nO, num_points, 3 OR
        nB, nO, num_points, 3
    :return:
    """
    max_num_points = -1
    max_num_objects = -1

    for arr in list_of_np_arrays:
        # assert len(arr.shape) <= 2, arr.shape
        if len(arr.shape) <= 2:
            point_idx = 0
        else:
            # len(arr.shape) == 3
            # each arr is [nO, num_points, 3]
            point_idx = 1
            object_idx = 0
            max_num_objects = max(max_num_objects, arr.shape[object_idx])
        max_num_points = max(max_num_points, arr.shape[point_idx])
    assert max_num_points > 0, max_num_points
    # assert max_num_objects > 0, max_num_objects
    for idx, arr in enumerate(list_of_np_arrays):
        if len(arr.shape) == 1:
            list_of_np_arrays[idx] = np.pad(list_of_np_arrays[idx], pad_width=((0,max_num_points - arr.shape[0])))
        elif len(arr.shape) == 2:
            list_of_np_arrays[idx] = np.pad(list_of_np_arrays[idx], pad_width=((0,max_num_points - arr.shape[0]), (0,0)))
        else:
            list_of_np_arrays[idx] = np.pad(list_of_np_arrays[idx], pad_width=((0, max_num_objects - arr.shape[0]), (0, max_num_points - arr.shape[1]), (0, 0)))
    return list_of_np_arrays


def load_hyperconfig_from_filepath(filepath):
    spec = importlib.util.spec_from_file_location("hc", filepath)
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    return hc.config


def load_ldd_function_from_filepath(filepath):
    spec = importlib.util.spec_from_file_location("lc", filepath)
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    return lc.get_loss_and_diag_dict


# def get_git_commit_from_disk():
#     git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode(("utf-8")).split("\n")[0]
