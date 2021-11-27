# CVAE with bottom_thresholded_boolean
import torch
from scipy.spatial.transform import Rotation as R
from utils.loss_util import CVAELoss

loss_params = dict(recon_loss_type="bce",
                   sigmoid_variance=2000,
                   sigmoid_offset=10,
                   kld_weight=[0, .02],
                   kl_type="mog_prior_kld",
                   prior_kld_weight=0,
                   positive_class_weight=2,
                   equalize_num_pos_num_neg=True,
                   iteration_count=600 * 100,
                   kld_lower=-10000000,
                   kld_upper=10000000
                   )


loss_fnx = CVAELoss(**loss_params)


def get_loss_and_diag_dict(predictions, batch, increment_iteration=False,
                                                                 loss_params=None):
    target = batch['bottom_thresholded_boolean']
    decoder_predictions = predictions['decoder'][0]
    loss, reconstruction_loss, kld_loss, kld_weight, extra_diag_dict = loss_fnx(decoder_predictions,
                                                                                target,
                                                                                canonical_pc=batch[
                                                                                    'canonical_pointcloud'],
                                                                                rotated_quat_matrices=R.from_quat(
                                                                                    batch['rotated_quat'].squeeze(
                                                                                        1)).as_matrix(),
                                                                                rotated_pc=batch['rotated_pointcloud'],
                                                                                **predictions['prior'],
                                                                                **predictions['encoder']
                                                                                )

    diag_dict = dict(reconstruction_loss=reconstruction_loss.data.cpu().numpy(),
                     kld_loss=kld_loss.data.cpu().numpy(),
                     kld_weight=kld_weight.data.cpu().numpy() if isinstance(kld_weight, torch.Tensor) else kld_weight
                     )

    diag_dict.update(extra_diag_dict)

    for k, v in diag_dict.items():
        if isinstance(v, torch.Tensor):
            diag_dict[k] = v.data.cpu().numpy()
    return loss, diag_dict