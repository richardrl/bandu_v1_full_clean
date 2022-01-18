import pybullet as p
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('urdf_path',
                    type=str,
                    help='URDF Path')

args = parser.parse_args()



p.connect(p.GUI)
obj_id = p.loadURDF(args.urdf_path, globalScaling=1)

# Bandu Block
# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi/2, np.pi/2]), 0.0, np.random.uniform(0, 2*np.pi)]).as_quat())

# Skewed Rectangular Prism
# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi, 0, np.pi]), np.random.choice([-np.pi, np.pi, 0]), np.random.uniform(0, 2*np.pi)]).as_quat())

# p.resetBasePositionAndOrientation(obj_id, [0.0,0.0,0.0], R.from_euler("xyz", [np.random.choice([-np.pi]), np.random.choice([-np.pi]), np.random.uniform(0, 2*np.pi)]).as_quat())

aabb = p.getAABB(obj_id)

print(aabb)

aabbMinVec = aabb[0]
aabbMaxVec = aabb[1]

print("obj height")
print(aabbMaxVec[-1] - aabbMinVec[-1])
while 1:
    pass