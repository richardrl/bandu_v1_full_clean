import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from supervised_training.models.pointnet_models import PointnetCls, PointnetSeg
from supervised_training.utils.vae_util import gumbel_softmax

class CVAECls(nn.Module):
    def __init__(self, embedding_dim, latent_dim, encoder_kwargs, decoder_kwargs):
        super().__init__()
        self.pointcloud_encoder = PointnetCls(output_dim=embedding_dim,
                                              normal_channel=False, **encoder_kwargs)
        self.fc_embedding2z = nn.Linear(embedding_dim, 2*latent_dim)
        self.fc_embedding2quat = nn.Linear(embedding_dim, 4)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.latent_to_quat_decoder = PointnetCls(output_dim=embedding_dim, normal_channel=False, **decoder_kwargs)

    def reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps * std + z_mu

    def forward(self, batch):
        # this is the input X
        nB, num_objects, num_points, _ = batch['rotated_pointcloud'].shape

        # -> nB, num_points, 3
        assert len(batch['rotated_pointcloud'].shape) == 4
        pc = batch['rotated_pointcloud'].squeeze(1)

        # # concatenate the label
        # # -> nB, num_points, 6
        # pc = torch.cat((pc, batch['pointcloud'].squeeze(1)), dim=-1)

        # concatenate the label
        # nB, num_points, 3 AND nB, 4 -> nB, num_points, 7
        prepped_quats = batch['quats_applied'].unsqueeze(1).expand(-1, num_points, -1)
        pc = torch.cat((pc, prepped_quats), dim=-1)

        embedding = self.pointcloud_encoder(pc.permute(0, 2, 1))

        # -> nB, embedding_dim and nB, embedding_dim
        z_mu, z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)

        # -> nB, latent_dim
        z_sample = self.reparameterize(z_mu, z_logvar)

        batch['encoder_z_sample'] = z_sample

        # nB, num_points, latent_dim
        z_expanded = z_sample.unsqueeze(1).expand(-1, num_points, -1)

        # Setup pc for decoder
        pc = torch.cat((batch['rotated_pointcloud'].squeeze(1), z_expanded), dim=-1).permute(0, 2, 1)
        # the decoder will concatenate encoder_z_sample to each point
        embedding = self.latent_to_quat_decoder(pc)

        # nB, 4
        pred_quat = self.fc_embedding2quat(embedding)

        pred_quat = pred_quat / torch.norm(pred_quat, p=2, dim=-1, keepdim=True)
        return pred_quat, z_mu, z_logvar


class CVAESegGumbel(nn.Module):
    def __init__(self, embedding_dim=None,
                 latent_dim=None,
                 categorical_dim=None,
                 encoder_kwargs=None,
                 decoder_kwargs=None,
                 temperature=None,
                 temperature_drop_idx=25000,
                 anneal_rate=0.00003,
                 temp_min = 0.5,
                 batch_idx=50000,
                 normalize=True,
                  multi_gpu=False):

        super().__init__()
        self.normalize = normalize
        self.pointcloud_encoder = PointnetCls(output_dim=embedding_dim, normal_channel=False, **encoder_kwargs)
        self.fc_embedding2categorical_logits = nn.Linear(embedding_dim, latent_dim * categorical_dim)
        self.fc_embedding2quat = nn.Linear(embedding_dim, 4)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # number of classes for each categorical variable
        self.categorical_dim = categorical_dim
        self.pointcloud_decoder = PointnetSeg(output_dim=1, normal_channel=False, **decoder_kwargs)

        # spreads out the subnetworks on gpu 1 and gpu 0
        self.multi_gpu = multi_gpu

        # initial start_temperature
        self.batch_idx = batch_idx
        # self.current_batch_idx = nn.Parameter(torch.Tensor([current_batch_idx]))
        self.temperature = temperature
        self.current_temp = temperature
        self.temperature_drop_idx = temperature_drop_idx
        self.anneal_rate = anneal_rate
        self.temp_min = temp_min

    def forward(self, batch):
        self.batch_idx += 1
        if self.multi_gpu:
            self.pointcloud_encoder.to(torch.device("cuda:1"))
            self.pointcloud_decoder.to(torch.device("cuda:0"))

        # this is the input X
        nB, num_objects, num_points, _ = batch['rotated_pointcloud'].shape

        # -> nB, num_points, 3
        assert len(batch['rotated_pointcloud'].shape) == 4
        pc = batch['rotated_pointcloud'].reshape(-1, num_points, 3)

        if self.normalize:
            # normalize
            # nB, num_points, 3
            pc = (pc - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)

        # # concatenate the label
        # # -> nB, num_points, 6
        # pc = torch.cat((pc, batch['pointcloud'].squeeze(1)), dim=-1)

        # concatenate the label (binary logit indices...)
        # nB, num_points, 3 AND nB, 4 -> nB, num_points, 7

        # prepped_quats = batch['quats_applied'].unsqueeze(1).expand(-1, num_points, -1)
        # pc = torch.cat((pc, prepped_quats), dim=-1)

        # batch['bottom_thresholded_boolean']: nB, nO, num_points, 1
        # -> nB*nO, num_points, -1
        prepped_bhb = (batch['bottom_thresholded_boolean'].reshape(-1, num_points, 1) - .5) * .01
        pc = torch.cat((pc, prepped_bhb), dim=-1)

        # before permutation, pc is nB*nO, num_points, 4
        # after permutation, it is nB*nO, 4, num_points, which is what is expected
        # -> nB*nO, 512
        embedding = self.pointcloud_encoder(pc.permute(0, 2, 1))

        # -> nB, embedding_dim and nB, embedding_dim
        z_logits = self.fc_embedding2categorical_logits(embedding).reshape(nB, self.latent_dim, self.categorical_dim)

        # -> nB, latent_dim
        if self.batch_idx < self.temperature_drop_idx:
            temp = self.temperature
        else:
            temp = np.maximum(self.temperature * np.exp(-self.anneal_rate * (self.batch_idx - self.temperature_drop_idx)), self.temp_min)

        # expose current_tmp for logging
        self.current_temp = temp
        z_sample, y_presoftmax_logits = gumbel_softmax(z_logits, temp, self.latent_dim, self.categorical_dim)

        # batch['encoder_z_sample'] = encoder_z_sample

        # nB, num_points, latent_dim
        # copy the global object level latent to each point
        z_expanded = z_sample.unsqueeze(1).expand(-1, num_points, -1)

        # Setup pc for decoder
        # Concatenate the z to every point
        # -> nB*nO, num_points, 3 X nB*nO, num_points, 4 -> nB*nO, 7, num_points
        pc = batch['rotated_pointcloud'].reshape(-1, num_points, 3)
        if self.normalize:
            # normalize
            # nB, num_points, 3
            pc = (pc - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)

        pc = torch.cat((pc, z_expanded), dim=-1).permute(0, 2, 1)
        # the decoder will concatenate encoder_z_sample to each point

        if self.multi_gpu:
            pc = pc.to(torch.device("cuda:0"))
        # -> nB*nO, num_points, 1 -> nB*nO, num_points
        pre_sigmoid_logits = self.pointcloud_decoder(pc).squeeze(-1)

        # ..., we represent the categorical distribution smoothly with a softmax over the logits
        return pre_sigmoid_logits, F.softmax(z_logits, dim=-1), y_presoftmax_logits

    def decode_batch(self, batch, std=1, ret_eps=False):
        if self.multi_gpu:
            self.pointcloud_decoder.to(torch.device("cuda:0"))
        # decodes batch by sampling and running through decoder
        nB, nO, num_points, _ = batch['rotated_pointcloud'].shape

        # sample_idx from iid gaussians
        # make the STD small here, improves results by a lot compared to std=1
        # eps = torch.randn(nB, num_points, self.latent_dim * self.categorical_dim).to(batch['rotated_pointcloud'].device) * std

        # uniform_logits = torch.ones((nB, num_points, self.latent_dim * self.categorical_dim))
        uniform_logits = torch.ones((nB*nO, self.latent_dim * self.categorical_dim))

        # -> nB*nO, self.latent_dim * self.categorical_dim
        eps = torch.distributions.one_hot_categorical.OneHotCategorical(logits=uniform_logits).sample()


        eps = eps.unsqueeze(1).expand(nB*nO, num_points, self.latent_dim * self.categorical_dim).to(batch['rotated_pointcloud'].device)
        print("ln461 eps")
        print(eps)

        pc = batch['rotated_pointcloud'].reshape(-1, num_points, 3)
        if self.normalize:
            # normalize
            # nB, num_points, 3
            pc = (pc - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)
        pc = torch.cat((pc, eps), dim=-1).permute(0, 2, 1)

        if self.multi_gpu:
            pc = pc.to(torch.device("cuda:0"))
        pre_sigmoid_logits = self.pointcloud_decoder(pc).squeeze(-1)

        if ret_eps:
            return pre_sigmoid_logits, eps
        else:
            return pre_sigmoid_logits


class CVAERegressPoint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass