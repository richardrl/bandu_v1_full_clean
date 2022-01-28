import torch
from models.generic_modules import ResidualBlock
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import ReversePermutation, MaskedAffineAutoregressiveTransform, CompositeTransform
from torch import nn, distributions as D
from torch.nn import functional as F

from utils import vae_util, transform_util, loss_util, train_util, pointcloud_util
import numpy as np
from models.dgcnn_partseg import DGCNNPartSeg
from models.dgcnn_cls import DGCNNCls
from models.dgcnn_cls_small import DGCNNClsSmall
from models.pointnet_models import PointnetCls, PointnetSeg, PointnetSegAndCls, PointnetClsAndCls

from scipy.spatial.transform import Rotation as R
from imports.bingham_rotation_learning.qcqp_layers import A_vec_to_quat


class DGCNNCVAESeg(nn.Module):
    def __init__(self,
                 embedding_dim,
                 latent_dim,
                 encoder_kwargs,
                 decoder_kwargs,
                 multi_gpu=False,
                 prior_type="normal",
                 prior_predictor_type="segncls",
                 prior_predictor_kwargs=dict(),
                 posterior_type="normal",
                 posterior_kwargs=dict(),
                 encoder_type="cls",
                 decoder_type="seg",
                 vamp_pseudo_components=50,
                 num_mog_prior_components=5,
                 lgmm_components=2,
                 lgmm_spacing=.5,
                 normalize=False,
                 logvar_upper_bound=.87,
                 logvar_lower_bound=-5,
                 use_normals=False,
                 label_type="btb",
                 svd_rotation_head=False,
                 residual_target_pc=True,
                 start_temperature=2,
                 temperature_drop_idx=25000,
                 anneal_rate=0.00003,
                 temp_min = 0.5,
                 current_batch_idx=0,
                 sample_prior_bool=False,
                 quaternion_head=False,
                 A_vec_to_quat_head=False,
                 gpu0=0,
                 gpu1=1,
                 use_fixed_temp=True):
        """

        :param embedding_dim:
        :param latent_dim:
        :param encoder_kwargs:
        :param decoder_kwargs:
        :param multi_gpu:
        :param prior_type:
        :param posterior_type:
        :param posterior_kwargs:
        :param vamp_pseudo_components:
        :param lgmm_components:
        :param lgmm_spacing: Determines how far apart the lgmm modes are
        :param normalize:
        """
        super().__init__()
        assert encoder_type in ["cls", "cls_small"]

        if encoder_type == "cls":
            self.pointcloud_encoder = DGCNNCls(
                                           **encoder_kwargs)
        else:
            # cls_small
            self.pointcloud_encoder = DGCNNClsSmall(
                **encoder_kwargs)
        self.fc_embedding2quat = nn.Linear(embedding_dim, 4)
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        assert decoder_type in ["seg", "cls", "segncls", "pointnet_seg"]
        self.decoder_type = decoder_type
        if self.decoder_type == "seg":
            self.pointcloud_decoder = DGCNNPartSeg(**decoder_kwargs)
        elif self.decoder_type == "cls":
            self.pointcloud_decoder = DGCNNCls(**decoder_kwargs)
        elif self.decoder_type == "pointnet_seg":
            self.pointcloud_decoder = PointnetSeg(normal_channel=False, **decoder_kwargs)
        else:
            self.pointcloud_decoder = PointnetSegAndCls(normal_channel=False, **decoder_kwargs)

        self.num_mog_prior_components = num_mog_prior_components


        self.logvar_upper_bound = logvar_upper_bound
        self.logvar_lower_bound = logvar_lower_bound

        # spreads out the subnetworks on gpu 1 and gpu 0
        self.multi_gpu = multi_gpu

        self.gpu_0 = torch.device(f"cuda:{gpu0}")
        self.gpu_1 = torch.device(f"cuda:{gpu1}")
        self.normalize = normalize

        self.posterior_type = posterior_type
        assert posterior_type in ["normal", "gmm", "realnvp", "maf"]

        if posterior_type == "normal" or posterior_type == "realnvp" or posterior_type == "maf":
            self.fc_embedding2z = nn.Linear(embedding_dim, 2*latent_dim)
        else:
            assert "vamp" in prior_type
            # posterior_type == "gmm"
            self.fc_embedding2z = nn.Linear(embedding_dim, 2*latent_dim*vamp_pseudo_components
                                            + vamp_pseudo_components)

        if posterior_type == "maf" or prior_type == "maf_predict":
            # num_layers = 5
            base_dist = StandardNormal(shape=[self.latent_dim])

            transforms = []

            num_l = posterior_kwargs['num_layers'] if "num_layers" in posterior_kwargs else prior_predictor_kwargs['num_layers']

            p_k = posterior_kwargs['maf_kwargs'] if "maf_kwargs" in posterior_kwargs else prior_predictor_kwargs['maf_kwargs']
            for _ in range(num_l):
                transforms.append(ReversePermutation(features=self.latent_dim))
                transforms.append(MaskedAffineAutoregressiveTransform(features=self.latent_dim,
                                                                      context_features=self.embedding_dim,
                                                                      **p_k))
            self.transform = CompositeTransform(transforms)
            self.flow = Flow(self.transform, base_dist, embedding_net=nn.Sequential(ResidualBlock(1024),
                                                                                    nn.Linear(1024, self.embedding_dim)))

            if prior_type == "maf_predict":
                # takes in pc_X and predicts
                self.prior_predictor = PointnetCls(output_dim=self.latent_dim * 2 * prior_predictor_kwargs['num_components'] + \
                                                              prior_predictor_kwargs['num_components'],
                                                   channel=6 if use_normals else 3,
                                                   normal_channel=use_normals)
                self.prior_components = prior_predictor_kwargs['num_components']
            self.flow.apply(train_util.init_weights)

        assert label_type in ["btb", "canonical_pc", "rotated_quat_inv"]
        self.label_type = label_type
        self.num_X_channels = 6 if use_normals else 3
        self.num_Y_channels = 3 if label_type == "canonical_pc" else 1

        assert prior_type in ["normal", "vamp", "vamp_weighted", "vamp_weighted_predict",
                              "lattice_gmm_2components",
                              "lattice_gmm_4components",
                              "mog_predict",
                              "maf_predict",
                              "mog_predict_unit_logvar"]
        self.prior_type = prior_type
        if prior_type == "vamp" or prior_type == "vamp_weighted":
            # Should be same size as pc before going into encoder
            # num_vamp_components, num_points, 4
            # making this 4D is INCORRECT! It should only be the size of the segmentation mask.
            self.pseudo_input_y = nn.Parameter(torch.Tensor(vamp_pseudo_components, 2048, 1))
            self.pseudo_input_y.data.uniform_(-3e-3, 3e-3)
            # self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
            # self.input_independent_query.data.uniform_(-init_w, init_w)
            self.vamp_pseudo_components = vamp_pseudo_components
            if prior_type == "vamp_weighted":
                self.vamp_weights = nn.Parameter(torch.zeros(vamp_pseudo_components))
            else:
                self.vamp_weights = torch.ones(vamp_pseudo_components)
        elif prior_type == "vamp_weighted_predict" or "mog_predict" in prior_type:
            assert prior_predictor_type in ["segncls", "clsncls", "cls", "cls_small"]
            self.prior_predictor_type = prior_predictor_type

            self.vamp_pseudo_components = vamp_pseudo_components

            if prior_predictor_type == "segncls":
                self.prior_predictor = PointnetSegAndCls(seg_output_dim=vamp_pseudo_components * self.num_Y_channels,
                                                         cls_output_dim=vamp_pseudo_components,
                                                         normal_channel=use_normals, channel=6 if use_normals else 3,
                                                         **prior_predictor_kwargs
                                                         )
            elif prior_predictor_type == "cls":
                # output dim should be:
                # mean, logvars, + categorical
                # num_components * latent_dim * 2 + num_components
                # self.prior_predictor = DGCNNCls(output_dim=num_mog_prior_components * self.latent_dim * 2 + num_mog_prior_components,
                #                                    channel=6 if use_normals else 3,
                #                                    normal_channel=use_normals)
                self.prior_predictor = DGCNNCls(n_knn=20,
                                       num_class=num_mog_prior_components * self.latent_dim * 2 + num_mog_prior_components,
                                                normal_channel=use_normals,
                                                **prior_predictor_kwargs)
            elif prior_predictor_type == "cls_small":
                # output dim should be:
                # mean, logvars, + categorical
                # num_components * latent_dim * 2 + num_components
                # self.prior_predictor = DGCNNCls(output_dim=num_mog_prior_components * self.latent_dim * 2 + num_mog_prior_components,
                #                                    channel=6 if use_normals else 3,
                #                                    normal_channel=use_normals)
                self.prior_predictor = DGCNNClsSmall(n_knn=20,
                                                num_class=num_mog_prior_components * self.latent_dim * 2 + num_mog_prior_components,
                                                normal_channel=use_normals)
            else:
                self.prior_predictor = PointnetClsAndCls()
        elif prior_type == "lattice_gmm_2components":
            assert lgmm_components == 2
            self.lgmm_components = lgmm_components
            self.lgmm_mu = torch.Tensor([[-1, -1, -1], [+1, +1, +1]]) * lgmm_spacing
            lgmm_vars = torch.Tensor([[.2, .2, .2], [.2, .2, .2]])
            self.lgmm_logvars = torch.log(lgmm_vars)
            self.lgmm_weights = torch.Tensor([1/lgmm_components, 1/lgmm_components])
        elif prior_type == "lattice_gmm_4components":
            assert lgmm_components == 4
            self.lgmm_components = lgmm_components
            self.lgmm_mu = torch.Tensor([[-1, -1, -1],
                                         [+1, +1, +1],
                                         [+1, -1, -1],
                                         [-1, +1, +1]]) * lgmm_spacing
            lgmm_vars = torch.ones_like(self.lgmm_mu)
            self.lgmm_logvars = torch.log(lgmm_vars)
            self.lgmm_weights = torch.Tensor([1/lgmm_components,
                                              1/lgmm_components,
                                              1/lgmm_components,
                                              1/lgmm_components])
        self.use_normals = use_normals
        self.svd_rotation_head = svd_rotation_head

        self.quaternion_head = quaternion_head

        self.A_vec_to_quat_head = A_vec_to_quat_head

        # whether or not to sum the decoder translations with the existing pointcloud
        self.residual_target_pc = residual_target_pc

        # for gumbel
        self.temperature = start_temperature
        self.current_temp = start_temperature
        self.temperature_drop_idx = temperature_drop_idx
        self.anneal_rate = anneal_rate
        self.temp_min = temp_min
        self.current_batch_idx = current_batch_idx
        self.sample_prior_bool = sample_prior_bool
        self.use_fixed_temp = use_fixed_temp

    def reparameterize(self, z_mu, z_logvar):
        # transforms individual independent normal samples by z_mu and z_logvar
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps * std + z_mu

    def forward(self, batch):
        self.current_batch_idx += 1

        if self.multi_gpu:
            self.pointcloud_encoder.to(self.gpu_1)

            self.fc_embedding2z.to(next(self.pointcloud_encoder.parameters()).device)

            self.pointcloud_decoder.to(self.gpu_0)

            if "predict" in self.prior_type:
                self.prior_predictor.to(self.gpu_0)

        # this is the input X
        nB, num_objects, num_points, _ = batch['rotated_pointcloud'].shape

        # -> nB, num_points, 3
        assert len(batch['rotated_pointcloud'].shape) == 4
        pc_X = batch['rotated_pointcloud'].reshape(-1, num_points, 3)

        if self.use_normals:
            # if this line fails, then that means the rotated normals is None
            normals_X = batch['rotated_normals'].reshape(-1, num_points, 3)

            if self.normalize:
                assert torch.all(batch['rotated_normals_var'] > 0)
                normals_X = (normals_X - batch['rotated_normals_mean'])/\
                            (torch.sqrt(batch['rotated_normals_var']) + 1E-6)

        if self.label_type == "btb":
            # batch['bottom_thresholded_boolean']: nB, nO, num_points, 1
            # -> nB*nO, num_points, -1
            # prepped_bhb is the Y mask... this is what we should learn as pseudoinputs for vampprior
            reshaped_label = (batch['bottom_thresholded_boolean'].reshape(-1, num_points, 1) - .5) * .01
        elif self.label_type == "rotated_quat_inv":
            inv_quat = torch.Tensor(R.from_quat(batch['rotated_quat'].cpu().data.numpy()).inv().as_quat())
            reshaped_label = inv_quat.unsqueeze(1).expand(nB, num_points, 4)
        elif self.label_type == "canonical_pc":
            # label_type == canonical_pc
            reshaped_label = batch['canonical_pointcloud'].reshape(-1, num_points, 3)
        else:
            raise NotImplementedError

        if self.normalize:
            # normalize
            # nB, num_points, 3
            assert torch.all(batch['rotated_pointcloud_var'] > 0), print("Did you load a stats.json?")

            pc_X = (pc_X - batch['rotated_pointcloud_mean'])/\
                   (torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)

            if self.label_type == "canonical_pc":
                reshaped_label = (reshaped_label - batch['canonical_pointcloud_mean'])/ \
                                 (torch.sqrt(batch['canonical_pointcloud_var']) + 1E-6)
            elif self.label_type == "rotated_quat_inv":
                reshaped_label = (reshaped_label - batch['rotated_quat_inv_mean']) / \
                                 (torch.sqrt(batch['rotated_quat_inv_var']) + 1E-6)
            else:
                # don't normalize the label...
                print("ln306 label type")
                print(self.label_type)
                # raise NotImplementedError

        if self.use_normals:
            pc = torch.cat((pc_X, normals_X, reshaped_label), dim=-1)
        else:
            pc = torch.cat((pc_X, reshaped_label), dim=-1)

        embedding = self.pointcloud_encoder(pc.permute(0, 2, 1))

        if self.posterior_type == "normal":
            # -> nB, latent_dim and nB, latent_dim
            encoder_z_mu, encoder_z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)

            encoder_z_logvar = torch.clamp(encoder_z_logvar, self.logvar_lower_bound, self.logvar_upper_bound)

            print("ln280 encoder z mu _ max component")
            print(encoder_z_mu)
            print("ln280 encoder z logvar _ max component")
            print(encoder_z_logvar)

            # -> nB, latent_dim
            encoder_z_sample = self.reparameterize(encoder_z_mu, encoder_z_logvar)
        elif self.posterior_type == "realnvp":
            encoder_z_mu, encoder_z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)
            # transform = transforms.CompositeTransform([
            #     transforms.AffineCouplingTransform(),
            #     transforms.RandomPermutation(features=2)
            # ])
        elif self.posterior_type == "maf":
            # encoder_z_mu, encoder_z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)
            # encoder_z_sample = self.reparameterize(encoder_z_mu, encoder_z_logvar)
            # std = torch.exp(0.5 * z_logvar)
            eps = torch.randn(nB, self.latent_dim).to(pc.device)

            eps = torch.clamp(eps, min=-3, max=+3)

            print("ln391 eps")
            print(eps)
            print("ln393 embedding nans")
            print(torch.sum(torch.isnan(embedding), -1))
            encoder_z_sample, logabsdet = self.flow._transform.inverse(eps, context=embedding)

            print("ln385 encoder_z_sample")
            print(encoder_z_sample)
            flow_log_prob = self.flow.log_prob(inputs=encoder_z_sample, context=embedding)
        else:
            raise NotImplementedError
        # else:
        #     # posterior_type == "gmm"
        #     # -> nB, vamp_pseudo_components, latent_dim and nB, vamp_pseudo_components, latent_dim
        #     encoder_z_mu, encoder_z_logvar, z_components = self.fc_embedding2z(embedding).split([self.latent_dim*self.vamp_pseudo_components,
        #                                                            self.latent_dim*self.vamp_pseudo_components,
        #                                                            self.vamp_pseudo_components], dim=-1)
        #
        #     mix = D.Categorical(logits=z_components)
        #     comp = D.Independent(D.Normal(loc=encoder_z_mu.reshape(nB, self.vamp_pseudo_components, self.latent_dim),
        #                                   scale=torch.exp(encoder_z_logvar.reshape(nB, self.vamp_pseudo_components, self.latent_dim))**(1/2)), 1)
        #     # comp = D.Normal(loc=gaussian_means, scale=torch.relu(gaussian_logstddevs) + 1E-11)
        #     gmm = D.MixtureSameFamily(mix, comp)

        """
        Decoder start
        """

        # nB, num_points, latent_dim
        z_expanded = encoder_z_sample.unsqueeze(1).expand(-1, num_points, -1)

        # Setup pc for decoder
        pc_X = batch['rotated_pointcloud'].reshape(-1, num_points, 3)

        if self.normalize:
            # normalize
            # nB, num_points, 3
            assert torch.all(batch['rotated_pointcloud_var'] > 0)
            pc_X = (pc_X - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)
        pc = pc_X

        if self.use_normals:
            normals_X = batch['rotated_normals'].reshape(-1, num_points, 3)

            if self.normalize:
                assert torch.all(batch['rotated_normals_var'] > 0)
                normals_X = (normals_X - batch['rotated_normals_mean'])/ \
                            (torch.sqrt(batch['rotated_normals_var']) + 1E-6)
            pc = torch.cat((pc_X, normals_X), dim=-1)

        pc = pc.to(next(self.parameters()).device)
        # Concatenate the z to every point
        # -> nB*nO, num_points, 3 concat nB*nO, num_points, 4 -> nB*nO, 7, num_points
        pc = torch.cat((pc.float(), z_expanded), dim=-1).permute(0, 2, 1)
        # the decoder will concatenate encoder_z_sample to each point

        # -> nB*nO, num_points, 1 -> nB*nO, num_points
        if self.multi_gpu:
            pc = pc.to(self.gpu_0)

        if self.decoder_type == "segncls":
            # decoder_logits: nB, output_dim, num_points, 1 -> squeeze last dim
            # sample_level_prediction: nB, 1
            decoder_logits, decoder_logvars = self.pointcloud_decoder(pc)

            decoder_logvars = torch.clamp(decoder_logvars, self.logvar_lower_bound, self.logvar_upper_bound)

            # -> nB, num_points, output_dim
            decoder_logits = decoder_logits.squeeze(-1).transpose(-1, -2)
        else:
            # nB, num_points
            decoder_logits = self.pointcloud_decoder(pc).squeeze(1)

        # optionally, predict a logvar per input X
        # TODO

        # for producing a rotation, we need to use SVD to solve for the rotation matrix here
        # potentially need to do weighted SVD

        out = dict()
        if self.posterior_type == "maf":
            # out['encoder'] = [flow_log_prob, encoder_z_sample]
            out['encoder'] = dict(
                encoder_flow_log_prob=flow_log_prob,
                encoder_z_sample=encoder_z_sample
            )
        else:
            # out['encoder'] = [encoder_z_mu, encoder_z_logvar, encoder_z_sample]
            out['encoder'] = dict(encoder_z_mu=encoder_z_mu,
                                  encoder_z_logvar=encoder_z_logvar,
                                  encoder_z_sample=encoder_z_sample)

        # if self.pointcloud_encoder.feat1.stn_transform:
        #     out['encoder']['encoder_trans'] = encoder_trans

        if self.svd_rotation_head:
            # for pc2pc, this is: nB, num_points, 3
            # -> nB, 3, num_points
            decoder_corresponding_translations = decoder_logits.transpose(-1, -2)

            # -> nB, 3, num_points
            A = batch['rotated_pointcloud'].squeeze(1).transpose(-1, -2).detach()

            # the decoder can either output the absolute position of points, or output the residual

            target_pc = decoder_corresponding_translations + A.to(decoder_corresponding_translations.device) \
                if self.residual_target_pc else decoder_corresponding_translations
            if self.decoder_type == "seg":
                out['decoder'] = [pointcloud_util.rot_from_correspondences(nB, target_pc, A), decoder_corresponding_translations]
            else:
                out['decoder'] = [pointcloud_util.rot_from_correspondences(nB, target_pc, A), decoder_logvars]
        elif self.quaternion_head:
            assert not self.A_vec_to_quat_head
            quaternions = decoder_logits / torch.linalg.norm(decoder_logits, dim=-1).unsqueeze(-1)

            # convert to rotations
            rot_mats = transform_util.torch_quat2mat(quaternions)

            out['decoder'] = [rot_mats]
        elif self.A_vec_to_quat_head:
            q = A_vec_to_quat(decoder_logits)

            rot_mats = transform_util.torch_quat2mat(q)

            out['decoder'] = [rot_mats]
        else:
            out['decoder'] = [decoder_logits]

        """
        Prior start
        """
        if self.prior_type == "vamp" or self.prior_type == "vamp_weighted" or \
                self.prior_type == "vamp_weighted_predict" or self.prior_type == "vamp_weighted_predict_maf":
            if "vamp_weighted_predict" in self.prior_type:
                if self.use_normals:
                    pc = torch.cat((pc_X, normals_X), dim=-1)


                else:
                    pc = pc_X

                if self.prior_predictor.stn_transform:
                    pseudos, cats, prior_trans = self.prior_predictor(pc.permute(0, 2, 1).to(self.gpu_0))
                else:
                    pseudos, cats = self.prior_predictor(pc.permute(0, 2, 1).to(self.gpu_0))

                # -> nB, num_components, num_Y_channels, num_points
                pseudos = pseudos.reshape(nB, self.vamp_pseudo_components, self.num_Y_channels, num_points)

                # -> nB, num_components, num_points, num_Y_channels
                pseudos = pseudos.transpose(-1, -2)
                # print("ln670 pseudos")
                # print(pseudos)

                new_pc = torch.cat((pc.unsqueeze(1).expand(-1, self.vamp_pseudo_components, -1, -1),
                                    pseudos.to(pc.device)), dim=-1)
                weights = cats
            else:
                # we can either use input-independent vamp pseudo and vamp weights, or predict them based off of X
                new_pc = torch.cat((pc_X.unsqueeze(1).expand(-1, self.vamp_pseudo_components, -1, -1),
                                    self.pseudo_input_y.unsqueeze(0).expand(nB, -1, -1, -1)), dim=-1)
                weights = self.vamp_weights

            # -> nB*num_components, 512

            if self.pointcloud_encoder.feat1.stn_transform:
                embedding, prior_encoder_trans = self.pointcloud_encoder(new_pc.reshape(nB*self.vamp_pseudo_components,
                                                                   num_points,
                                                                   self.num_X_channels + self.num_Y_channels).transpose(-1, -2))
            else:
                embedding = self.pointcloud_encoder(new_pc.reshape(nB*self.vamp_pseudo_components,
                                                                   num_points,
                                                                   self.num_X_channels + self.num_Y_channels).transpose(-1, -2))

            # nB*num_components, 512 -> nB, num_components, 512
            embedding = embedding.reshape(nB, self.vamp_pseudo_components, 512)


                    # embedding = self.pointcloud_encoder(self.pseudo_input_y.permute(0, 2, 1))
            # -> 2X: nB, num_components, latent_dim
            vamp_z_mu, vamp_z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)

            vamp_z_logvar = torch.clamp(vamp_z_logvar, self.logvar_lower_bound, self.logvar_upper_bound)

            print("\n\n")
            print("ln713 categorical weights")
            print(weights)
            # out['prior'] = [vamp_z_mu, vamp_z_logvar, weights]
            out['prior'] = dict(prior_mog_z_mu=vamp_z_mu,
                                prior_mog_z_logvar=vamp_z_logvar,
                                prior_mog_weights=weights
                                )
        elif "lattice_gmm" in self.prior_type:
            out['prior'] = self.lgmm_mu.to(next(self.pointcloud_encoder.parameters()).device), \
                           self.lgmm_logvars.to(next(self.pointcloud_encoder.parameters()).device)
        elif "mog_predict" in self.prior_type:
            # take in pointcloud
            # predict fixed sized embedding
            # from that embedding, predict slab, which gets reshaped to be component means and variances,
            # and categorical weights
            # send these three elements to be the prior key of out dict

            z_mu, z_logvar, cat = self.mog_predict_prior_predict(pc_X, normals_X if self.use_normals else None)

            # out['prior'] = [z_mu, z_logvar, cat]
            out['prior'] = dict(
                prior_mog_z_mu=z_mu,
                prior_mog_z_logvar=z_logvar,
                prior_mog_weights=cat
            )
        elif "maf_predict" in self.prior_type:
            # first, given X, predict p(z|X) where z is reparameterized Gaussian
            # then, fit the base distribution to z
            # then, setup the flow
            # TODO: i don't think this log prob is calculated correctly
            if self.use_normals:
                pc = torch.cat((pc_X, normals_X), dim=-1)
            else:
                pc = pc_X

            z_params, global_feat = self.prior_predictor(pc.permute(0, 2, 1).to(self.gpu_0),
                                                         ret_global_feat=True)

            z_params = z_params.to(encoder_z_sample.device)
            z_categorical_components = z_params[:, -self.prior_components:]

            z_params = z_params[:, :-self.prior_components].reshape(nB, self.prior_components, self.latent_dim * 2)
            z_mu, z_logvar = z_params.split([self.latent_dim, self.latent_dim], dim=-1)

            mix = D.Categorical(logits=z_categorical_components)

            comp = D.Independent(D.Normal(loc=z_mu.reshape(nB, self.prior_components, self.latent_dim),
                                          scale=torch.exp(z_logvar.reshape(nB, self.prior_components, self.latent_dim))**(1/2)), 1)
            # comp = D.Normal(loc=gaussian_means, scale=torch.relu(gaussian_logstddevs) + 1E-11)
            gmm = D.MixtureSameFamily(mix, comp)

            del self.flow._distribution
            # self.flow._distribution = torch.distributions.MultivariateNormal(loc=z_mu.to(encoder_z_sample.device),
            #                                                                  scale_tril=torch.diag_embed(torch.exp(z_logvar.to(encoder_z_sample.device))**(1/2)))
            self.flow._distribution = gmm

            # the flow should be conditioned on the embedding X...
            # TODO: check how this context is actually used

            # global_feat: nB, 1024 shape
            flow_log_prob = self.flow.log_prob(inputs=encoder_z_sample, context=global_feat.to(encoder_z_sample.device))

            # out['prior'] = [flow_log_prob, z_categorical_components, z_mu, z_logvar]
            out['prior'] = dict(
                flow_log_prob=flow_log_prob,
                prior_mog_z_mu=z_mu,
                prior_mog_z_logvar=z_logvar,
                prior_mog_weights=z_categorical_components
            )
        else:
            out['prior'] = dict()
            # raise NotImplementedError

        # if "predict" in self.prior_type and self.prior_predictor.feat1.stn_transform:
        #     out['prior']['prior_trans'] = prior_trans

        """
        Prior sampling for forward KL
        
        Sample from prior, then feed this into the posterior and prior to get new likelihoods for forward KL
        """

        if self.sample_prior_bool:
            # -> nB, 1, latent_dim -> nB, latent_dim
            out['prior']['prior_z_sample'] = self.sample_prior(batch, 1, reparameterize=True).squeeze(1)
        return out

            # return decoder_pre_sigmoid_logits, encoder_z_mu, encoder_z_logvar

    def mog_predict_prior_predict(self, pc_X, normals_X=None):
        if self.use_normals:
            pc = torch.cat((pc_X, normals_X), dim=-1)
        else:
            pc = pc_X

        pp_device = next(self.prior_predictor.parameters()).device
        unshaped = self.prior_predictor(pc.permute(0, 2, 1).to(pp_device))

        meanlogvardim = self.num_mog_prior_components * self.latent_dim * 2
        catdim = self.num_mog_prior_components

        meanlogvar, cat = unshaped.split([meanlogvardim, catdim], dim=-1)

        nB = pc_X.shape[0]
        meanlogvar = meanlogvar.reshape(nB, self.num_mog_prior_components, self.latent_dim * 2)

        z_mu, z_logvar = meanlogvar.split([self.latent_dim, self.latent_dim], dim=-1)

        if "unit_logvar" in self.prior_type:
            z_logvar = torch.zeros_like(z_logvar)
        return z_mu, z_logvar, cat

    def sample_mog(self, weights, z_mu, z_logvar, num_components,
                   nB, nO, num_points, z_samples_per_sample=1, use_mean=False,
                   reparameterize=False):
        """

        Args:
            weights:
            z_mu:
            z_logvar:
            num_components:
            nB:
            nO:
            num_points:
            z_samples_per_sample:
            use_mean: Don't sample, just use the mean
            reparameterize:

        Returns:

        """
        nB = z_mu.shape[0]
        if reparameterize:
            assert z_samples_per_sample == 1, "z_samples_per_sample not implemented here yet"
            # start_temperature = .5

            if self.use_fixed_temp:
                temp_to_use = self.current_temp
            else:
                if self.current_batch_idx < self.temperature_drop_idx:
                    temp_to_use = self.temperature
                else:
                    temp_to_use = np.maximum(self.temperature * np.exp(-self.anneal_rate * (self.current_batch_idx - self.temperature_drop_idx)), self.temp_min)

                # expose current_tmp for logging
                self.current_temp = temp_to_use

            latent_dim = 1
            assert len(weights.shape) == 2
            categorical_dim = weights.shape[-1]

            # y_hard_mask: nB, num_cats
            y_hard_mask, y_pre_softmax_logits = \
                vae_util.gumbel_softmax(weights, temp_to_use, latent_dim, categorical_dim, hard=True)

            vamp_z_mu = z_mu.unsqueeze(1).expand(-1, z_samples_per_sample, -1, self.latent_dim)
            vamp_z_logvar = z_logvar.unsqueeze(1).expand(-1, z_samples_per_sample, -1, self.latent_dim)

            # -> nB, z_samples_per_sample, num_cats, latent_dim
            prior_sampled_z = self.reparameterize(vamp_z_mu,
                                                  vamp_z_logvar)

            # multiply by the categorical mask
            # -> nB, z_samples_per_sample, num_cats, latent_dim
            masked_z = y_hard_mask.unsqueeze(1).unsqueeze(-1).expand(-1, z_samples_per_sample, -1, self.latent_dim) \
                       * prior_sampled_z

            # reduce along the num_cats dim
            # -> nB, z_samples_per_sample, latent_dim
            prior_sampled_z = masked_z.sum(2)
        else:
            probs = F.softmax(weights.unsqueeze(1).expand(-1, z_samples_per_sample, -1), dim=-1)

            # nB, num_z_samples, num_components -> nB*num_z_samples, num_components
            probs = probs.reshape(-1, num_components)

            # indices = torch.argmax(probs, dim=-1, keepdim=True).to(z_mu.device)

            # # -> nB*num_z_samples, num_components ->  nB*num_z_samples -> nB, num_z_samples
            indices = torch.multinomial(probs, 1, replacement=True).reshape(nB, z_samples_per_sample).to(z_mu.device)

            vamp_z_mu = z_mu.unsqueeze(1).expand(-1, z_samples_per_sample, -1, self.latent_dim)
            indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.latent_dim)
            vamp_z_logvar = z_logvar.unsqueeze(1).expand(-1, z_samples_per_sample, -1, self.latent_dim)

            if use_mean:
                # -> nB, z_samples_per_sample, 1, latent_dim
                prior_sampled_z = torch.gather(vamp_z_mu,2,indices)
            else:
                # -> nB, z_samples_per_sample, 1, latent_dim
                prior_sampled_z = self.reparameterize(torch.gather(vamp_z_mu,
                                                                   2,
                                                                   indices),
                                                      torch.gather(vamp_z_logvar,
                                                                   2,
                                                                   indices))
        return prior_sampled_z

    def sample_prior(self, batch, z_samples_per_sample=1,
                     reparameterize=False,
                     use_mean=False,
                     **kwargs):
        """
        TODO make categorical distribution reparameterizable. need to take a start_temperature as input
        :param batch:
        :param z_samples_per_sample:
        :param kwargs:
        :return:
        """
        if self.multi_gpu:
            if self.prior_type == "vamp_weighted_predict":
                self.prior_predictor.to(self.gpu_0)

        # decodes batch by sampling and running through decoder
        nB, nO, num_points, _ = batch['rotated_pointcloud'].shape
        # -> nB*nO, num_points, 3
        pc_X = batch['rotated_pointcloud'].reshape(-1, num_points, 3)
        if self.normalize:
            # normalize
            # nB, num_points, 3
            pc_X = (pc_X - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)

        if self.prior_type == "normal":
            # make the STD small here, improves results by a lot compared to std=1
            std = 1

            # eps = torch.randn(nB * nO, num_points, self.latent_dim).to(batch['rotated_pointcloud'].device) * std
            prior_sampled_z = torch.randn(nB * nO, self.latent_dim) * std
            prior_sampled_z = prior_sampled_z.unsqueeze(1).expand(nB*nO, num_points, self.latent_dim).to(batch['rotated_pointcloud'].device)

            # print("prior_sampled_z")
            # print(prior_sampled_z)
        elif self.prior_type == "vamp" or self.prior_type == "vamp_weighted" or self.prior_type == "vamp_weighted_predict":
            if self.prior_type == "vamp_weighted_predict":
                pseudos, cats = self.prior_predictor(pc_X.permute(0, 2, 1).to(self.gpu_0))[:2]


                # -> nB, num_components, num_Y_channels, num_points
                pseudos = pseudos.reshape(nB, self.vamp_pseudo_components, self.num_Y_channels, num_points)

                # -> nB, num_components, num_points, num_Y_channels
                pseudos = pseudos.transpose(-1, -2)

                # nB, num_components, num_points, 3 +num_Y_channels
                new_pc = torch.cat((pc_X.unsqueeze(1).expand(-1, self.vamp_pseudo_components, -1, -1),
                                    pseudos.to(pc_X.device)), dim=-1)
                weights = cats

                if self.pointcloud_encoder.feat1.stn_transform:
                    embedding, encoder_trans = self.pointcloud_encoder(new_pc.reshape(nB*self.vamp_pseudo_components,
                                                                                      num_points,
                                                                                      3 + self.num_Y_channels).transpose(-1, -2))
                else:
                    embedding = self.pointcloud_encoder(new_pc.reshape(nB*self.vamp_pseudo_components,
                                                                       num_points,
                                                                       3 + self.num_Y_channels).transpose(-1, -2))
            else:
                embedding = self.pointcloud_encoder(self.pseudo_input_y.permute(0, 2, 1))
                weights = self.vamp_weights


            # -> 2x: nB*num_components, embedding_size
            vamp_z_mu, vamp_z_logvar = self.fc_embedding2z(embedding).split([self.latent_dim, self.latent_dim], dim=-1)
            vamp_z_logvar = torch.clamp(vamp_z_logvar, self.logvar_lower_bound, self.logvar_upper_bound)

            # vamp_z_mu: nB*num_components, num_latents
            # reshape it and make sure we are indexing into num_components dimension
            vamp_z_mu = vamp_z_mu.reshape(nB, self.vamp_pseudo_components, self.latent_dim)
            vamp_z_logvar = vamp_z_logvar.reshape(nB, self.vamp_pseudo_components, self.latent_dim)

            # sample_idx the categorical indices
            if "vamp_weighted" in self.prior_type:
                w = F.softmax(weights, dim=0) # K x 1 x 1
            else:
                w = torch.ones(self.vamp_pseudo_components)

            prior_sampled_z = self.sample_mog(w, vamp_z_mu, vamp_z_logvar,
                                              self.vamp_pseudo_components, nB, nO, num_points,
                                              z_samples_per_sample,
                                              use_mean=use_mean,
                                              reparameterize=reparameterize).to(batch['rotated_pointcloud'].device)

        elif self.prior_type == "lattice_gmm_4components":
            indices = torch.multinomial(torch.ones(self.lgmm_components), nB, replacement=True)
            std = 1
            eps = torch.randn(nB * nO, self.latent_dim) * std

            # -> nB, self.latent_dim
            prior_sampled_z = self.lgmm_mu[indices] + eps.to(self.lgmm_mu.device) * torch.exp(self.lgmm_logvars[indices])**(1/2)

            # -> nB, num_points, self.latent_dim
            # prior_sampled_z = prior_sampled_z.unsqueeze(1). \
            #     expand(nB*nO, num_points, self.latent_dim).to(batch['rotated_pointcloud'].device)
        elif self.prior_type == "maf_predict":
            if self.use_normals:
                # pc = torch.cat((pc_X, normals_X), dim=-1)
                raise NotImplementedError
            else:
                pc = pc_X

            z_params, global_feat = self.prior_predictor(pc.permute(0, 2, 1).to(self.gpu_1),
                                                         ret_global_feat=True)

            # z_mu, z_logvar = z_params.split([self.latent_dim, self.latent_dim], dim=-1)
            #
            # del self.flow._distribution
            # self.flow._distribution = torch.distributions.MultivariateNormal(loc=z_mu,
            #                                                                  scale_tril=torch.diag_embed(torch.exp(z_logvar)**(1/2)))
            z_categorical_components = z_params[:, -self.prior_components:]

            z_params = z_params[:, :-self.prior_components].reshape(nB, self.prior_components, self.latent_dim * 2)
            z_mu, z_logvar = z_params.split([self.latent_dim, self.latent_dim], dim=-1)

            mix = D.Categorical(logits=z_categorical_components)

            comp = D.Independent(D.Normal(loc=z_mu.reshape(nB, self.prior_components, self.latent_dim),
                                          scale=torch.exp(z_logvar.reshape(nB, self.prior_components, self.latent_dim))**(1/2)), 1)
            # comp = D.Normal(loc=gaussian_means, scale=torch.relu(gaussian_logstddevs) + 1E-11)
            gmm = D.MixtureSameFamily(mix, comp)

            del self.flow._distribution
            # self.flow._distribution = torch.distributions.MultivariateNormal(loc=z_mu.to(encoder_z_sample.device),
            #                                                                  scale_tril=torch.diag_embed(torch.exp(z_logvar.to(encoder_z_sample.device))**(1/2)))
            self.flow._distribution = gmm

            # self.flow._distribution = StandardNormal(loc=z_mu,
            #                                                                  scale_tril=torch.diag_embed(torch.exp(z_logvar)**(1/2)))
            # samples, log_prob = self.flow.sample_and_log_prob(1, context=global_feat)

            eps = self.flow._distribution.sample(torch.Size([]))

            embedded_context = self.flow._embedding_net(global_feat)

            prior_sampled_z, logabsdet = self.flow._transform.inverse(eps, context=embedded_context)
            # samples = self.flow._sample(nB, global_feat)

            # prior_sampled_z = prior_sampled_z.unsqueeze(1). \
            #     expand(nB*nO, num_points, self.latent_dim).to(batch['rotated_pointcloud'].device)
        elif "mog_predict" in self.prior_type:
            z_mu, z_logvar, cat = self.mog_predict_prior_predict(pc_X)
            w = F.softmax(cat, dim=0) # K x 1 x 1
            prior_sampled_z = self.sample_mog(w, z_mu, z_logvar, self.num_mog_prior_components, nB, 1, num_points,
                                              reparameterize=reparameterize)
        else:
            # normals
            pass
        return prior_sampled_z

    def decode_batch(self, batch, z_samples_per_sample=1, **kwargs):
        """

        :param batch:
        :param z_samples_per_sample: Mainly used for marginal likelihood calculation
        :param kwargs:
        :return:
        """

        if self.multi_gpu:
            self.pointcloud_decoder.to(self.gpu_0)
        nB, nO, num_points, channel = batch['rotated_pointcloud'].shape
        pc_X = batch['rotated_pointcloud'].reshape(-1, num_points, 3)

        if self.normalize:
            # normalize
            # nB, num_points, 3
            pc_X = (pc_X - batch['rotated_pointcloud_mean'])/(torch.sqrt(batch['rotated_pointcloud_var']) + 1E-6)

        if self.label_type == "btb":
            prior_sampled_z = self.sample_prior(batch, z_samples_per_sample,
                                                use_mean=True,
                                                **kwargs)
            prior_sampled_z = prior_sampled_z. \
                    expand(nB*nO, z_samples_per_sample, num_points, self.latent_dim)
        else:
            # prior_sampled_z = torch.randn(nB*nO, num_points, self.latent_dim).to(next(self.parameters()).device)
            prior_sampled_z = torch.zeros(nB*nO, num_points, self.latent_dim).to(next(self.parameters()).device)

        # prior_sampled_z = prior_sampled_z.unsqueeze(1). \
        #     expand(nB*nO, num_points, self.latent_dim).to(batch['rotated_pointcloud'].device)

        pc_X = pc_X.unsqueeze(1).expand(-1, z_samples_per_sample, -1, -1)\
            .reshape(nB*nO*z_samples_per_sample, num_points, -1).to(next(self.parameters()).device)

        pc = torch.cat((pc_X.float(),
                        prior_sampled_z.reshape(nB*nO*z_samples_per_sample, num_points, self.latent_dim)), dim=-1).permute(0, 2, 1)

        if self.multi_gpu:
            pc = pc.to(self.gpu_0)

        pre_sigmoid_logits = self.pointcloud_decoder(pc).squeeze(-1)

        if self.label_type == "rotated_quat_inv" or self.label_type == "canonical_pc":
            return pre_sigmoid_logits
        else:
            return pre_sigmoid_logits.reshape(nB*nO, z_samples_per_sample, num_points, -1)