import pickle
from utils import camera_util

cameras = camera_util.setup_cameras(dist_from_eye_to_focus_pt=.1,
                                    camera_forward_z_offset=-.2,
                                    cam_pkls="cameras/*.pkl")


for cam_idx, camera in enumerate(cameras):
    with open(f"out/{cam_idx}_cam.pkl", "wb") as fp:
        pickle.dump(dict(
            cam_ext_mat=camera.cam_ext_mat,
            cam_int_mat=camera.cam_int_mat
        ), fp)

