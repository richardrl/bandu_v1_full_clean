import pickle
from utils import camera_util
from bandu.config import BANDU_ROOT
import pathlib

cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                    camera_forward_z_offset=-.2,
                                    cam_pkls=BANDU_ROOT / "cameras/*.pkl")

for cam_idx, camera in enumerate(cameras):
    print(camera.cam_name)
    pathlib.Path(str(BANDU_ROOT / "out" / "cameras")).mkdir(parents=True, exist_ok=True)

    with open(str(BANDU_ROOT / "out" / "cameras" / f"{camera.cam_name}.pkl"), "wb") as fp:
        pickle.dump(dict(
            cam_ext_mat=camera.cam_ext_mat,
            cam_int_mat=camera.cam_int_mat
        ), fp)

