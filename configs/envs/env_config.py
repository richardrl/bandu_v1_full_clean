import pybullet as p
from bandu.config import BANDU_ROOT
env_params = dict(
    reward_type="height",
    # urdfs =["parts/urdfs/Colored_Block/Colored_Block.urdf_path",
    #         "parts/urdfs/Skewed Rectangular Prism/Skewed Rectangular Prism.urdf_path",
    #         "parts/urdfs/Pencil/Pencil.urdf_path",
    #         "parts/urdfs/Skewed Cylinder/Skewed Cylinder.urdf_path",
    #         "parts/urdfs/Skewed Triangular Prism/Skewed Triangular Prism.urdf_path",
    #         "parts/urdfs/Skewed Wedge/Skewed Wedge.urdf_path"],
    # stls=["parts/stls/Colored Block.STL",
    #       "parts/stls/Skewed Rectangular Prism.STL",
    #       "parts/stls/Pencil.STL",
    #       "parts/stls/Skewed Cylinder.STL",
    #       "parts/stls/Skewed Triangular Prism.STL",
    #       "parts/stls/Skewed Wedge.STL"],
    urdf_dir =str(BANDU_ROOT / "parts/vertical_urdfs/"),
    stl_dir=str(BANDU_ROOT / "parts/vertical_stls/"),
    # urdf_dir ="/home/richard/improbable/spinningup/parts/urdfs/engmikedset1",
    # stl_dir="/home/richard/improbable/spinningup/parts/stls/engmikedset1",
    # urdfs=["/home/richard/improbable/bandu_code/bandu/parts/vertical_urdfs/Skewed Rectangular Prism/Skewed Rectangular Prism.urdf_path"],
    # stls=["/home/richard/improbable/bandu_code/bandu/parts/vertical_stls/Skewed Rectangular Prism.stl"],
    # start_rotation_type="identity",
    start_rotation_type="full",
    urdf_holdout_frac=0, #what percentage of all urdfs to train on
    phase="train",
    p_connect_type=p.DIRECT,
    num_sub_steps=10,
    forward_sim_steps=500
)