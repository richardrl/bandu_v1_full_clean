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