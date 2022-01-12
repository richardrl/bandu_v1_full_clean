# load URDFs and get the canonical height for eachaabb = p.getAABB(obj_id)

# load directory

# for each episode, if the stable height is higher than the canonical height, we will treat it as a success

# otherwise, it will be a failure

# this corresponds with royal-brook-1975_checkpoint50
import sys
import pybullet as p
import glob

from pathlib import Path

from bandu.utils import bandu_util
import os
from supervised_training.utils.pb_spinningup_util import get_object_height

from bandu.config import TABLE_HEIGHT

import numpy as np


def stddev(binary_list_of_success_failures):
    import pdb
    pdb.set_trace()
    binary_list_of_success_failures = np.array(binary_list_of_success_failures)

    # calculate the mean
    mean = np.sum(binary_list_of_success_failures) / len(binary_list_of_success_failures)

    var = np.sum((binary_list_of_success_failures - mean)**2)/(len(binary_list_of_success_failures) - 1)
    return np.sqrt(var)


urdf_dir = sys.argv[1]
results_dir = sys.argv[2]

results_dir_basename = os.path.basename(os.path.normpath(results_dir))

max_trials = int(sys.argv[3]) # maximum number of trials before trials get ignored

# urdf_dir = "/home/richard/data/engmikedset/urdfs"
urdfs = [f for f in glob.glob(str(Path(urdf_dir) / "**/*.urdf"), recursive=True)]
urdfs += [f for f in glob.glob(str(Path(urdf_dir) / "**/*.URDF"), recursive=True)]

# mapping name to height
height_dict = dict()
best_theta_list_dict = dict()

results_save_dir = "/home/richard/Desktop/bandu_results/results dicts"

p.connect(p.DIRECT)
for urdf_path in urdfs:
    print("ln34 urdf_path")
    print(urdf_path)
    obj_id = p.loadURDF(urdf_path, globalScaling=1.5)

    # aabb = p.getAABB(obj_id)
    #
    # aabbMinVec = aabb[0]
    # aabbMaxVec = aabb[1]
    #
    # obj_height = aabbMaxVec[-1] - aabbMinVec[-1]
    obj_height = get_object_height(obj_id)

    object_name = bandu_util.parse_urdf_xml_for_object_name(os.path.dirname(urdf_path) + "/model.config")

    height_dict[object_name] = obj_height

import json

with open("height_dict.json", "w") as fp:
    json.dump(height_dict, fp, indent=4)

# results_dir = "/home/richard/data/results/engmikedset_dgcnn_mog5_unitvar"

height_success_dict = dict()
num_stacked_success_dict = dict()

json_paths = [f for f in glob.glob(str(Path(results_dir) / "**/*.json"), recursive=True)]

step_1_json_paths = [f for f in json_paths if "_1_" in f]

height_object_successes_dict = dict()
total_trials_dict = dict()

num_stacked_successes_dict = dict()


for k in height_dict.keys():
    height_object_successes_dict[k] = 0
    total_trials_dict[k] = 0
    best_theta_list_dict[k] = []
    num_stacked_successes_dict[k] = 0

import torch

for path in sorted(step_1_json_paths):
    # extract object names
    objects_folder_name = os.path.basename(os.path.dirname(path))

    non_foundation_obj_name = objects_folder_name.split("_foundation")[0]

    obj_height = height_dict[non_foundation_obj_name]

    with open(path, "r") as fp:
        dic = json.load(fp)

    urdf_path = Path("/home/richard/improbable/spinningup/parts/urdfs/main/engmikedset") / non_foundation_obj_name

    if "best_theta" not in dic.keys():
        best_theta_list_dict[non_foundation_obj_name].append(99)
    else:
        if np.isnan(dic['best_theta']):
            continue
        best_theta_list_dict[non_foundation_obj_name].append(dic['best_theta'])
    # pkl = torch.load(Path(os.path.dirname(path)) / f"ep0_2_o3d.pkl")
    #
    # assert pkl['batch']['pybullet_object_ids'][0] == 2
    #
    # # multiply relrot to get the target pose
    # # apply target pose to the normal vectors
    # pkl['batch']['current_quats']

    recorded_tower_height = dic['tower_height'] - TABLE_HEIGHT

    gt_tower_height = obj_height + height_dict['foundation']

    if total_trials_dict[non_foundation_obj_name] < max_trials:
        if recorded_tower_height > gt_tower_height * .98:
            height_object_successes_dict[non_foundation_obj_name] += 1

        assert dic['num_stacked'] <= 2, dic['num_stacked']
        if dic['num_stacked'] == 2:
            num_stacked_successes_dict[non_foundation_obj_name] += 1
        total_trials_dict[non_foundation_obj_name] += 1

best_theta_means_dict = dict()
best_theta_stddevs_dict = dict()

for obj_name in total_trials_dict.keys():
    # each obj name should have exactly max trials
    assert total_trials_dict[obj_name] == max_trials, total_trials_dict[obj_name]

for obj_name in height_dict.keys():
    height_success_dict[obj_name] = height_object_successes_dict[obj_name] / total_trials_dict[obj_name]
    num_stacked_success_dict[obj_name] = num_stacked_successes_dict[obj_name] / total_trials_dict[obj_name]
    best_theta_means_dict[obj_name] = np.mean(best_theta_list_dict[obj_name])
    best_theta_stddevs_dict[obj_name] = np.std(best_theta_list_dict[obj_name])
    assert num_stacked_successes_dict[obj_name] >= height_success_dict[obj_name]

# for object_name in height_dict.keys():
#     obj_dir = Path(results_dir) / object_name
#
#     json_paths = [f for f in glob.glob(str(Path(obj_dir) / "**/*.json"), recursive=True)]
#
#     object_successes = 0
#     object_total = 0
#
#     for pth in json_paths:
#         with open(pth, "r") as fp:
#             dic = json.load(fp)
#
#         logged_obj_height = float(dic['tower_height']) - TABLE_HEIGHT
#
#         if logged_obj_height > height_dict[object_name] * .98:
#             object_successes += 1
#         object_total += 1
#
#     height_success_dict[object_name] = object_successes/object_total

import pprint

rdb = Path(results_save_dir) / results_dir_basename
rdb.mkdir(exist_ok=True, parents=True)


with open(str(Path(results_save_dir) / results_dir_basename / f"maxtrials{max_trials}_height_success_means_dict.json"), "w") as fp:
    json.dump(height_success_dict, fp, indent=4)

with open(str(Path(results_save_dir) / results_dir_basename / f"maxtrials{max_trials}_num_stacked_success_dict.json"), "w") as fp:
    json.dump(num_stacked_success_dict, fp, indent=4)

with open(str(Path(results_save_dir) / results_dir_basename / f"maxtrials{max_trials}_best_theta_means_dict.json"), "w") as fp:
    json.dump(best_theta_means_dict, fp, indent=4)

with open(str(Path(results_save_dir) / results_dir_basename / f"maxtrials{max_trials}_best_theta_stddevs_dict.json"), "w") as fp:
    json.dump(best_theta_stddevs_dict, fp, indent=4)

print("Height success dict")
pprint.pprint(height_success_dict)

print("Best theta means dict")
pprint.pprint(best_theta_means_dict)

print("Best theta stddevs dict")
pprint.pprint(best_theta_stddevs_dict)
# pprint.pprint(num_stacked_success_dict)

# print("num trials total")
# # pprint.pprint(len(step_1_json_paths))
# pprint.pprint(total_trials_dict)