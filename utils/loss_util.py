import torch
import torch.nn as nn

def KL_divergence_gaussians_standard_prior(mu, logvar):
    # calculates kl divergence between normal gaussian and the learned encoder gaussian
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # logvar = log(sigma_2/sigma_1) where sigma_1 = 1
    # mu = u1-u2 where u2 = 0
    #
    res = -.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)

    # std = torch.exp(0.5 * logvar)
    #
    # res  = torch.rotated_pc_mean(torch.sum(- torch.log(std) + logvar.exp()*.5 + .5*mu**2 - .5, dim=-1))
    return res


def KL_divergence_independent_multivariate_gaussians(mu_q, mu_p, logvar_q, logvar_p):
    assert len(logvar_q.shape) == 2
    if len(logvar_p.shape) == 3:
        logvar_p = logvar_p.squeeze(1)
        mu_p = mu_p.squeeze(1)
    else:
        assert len(logvar_p.shape) == 2

    mu_p = mu_p.to(mu_q.device)
    logvar_p = logvar_p.to(logvar_q.device)

    #https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    Sigma_q = torch.diag_embed(torch.exp(logvar_q))
    Sigma_q_inv = torch.linalg.inv(Sigma_q)

    Sigma_p = torch.diag_embed(torch.exp(logvar_p))
    term1 = torch.sum(logvar_q, dim=-1) - torch.sum(logvar_p, dim=-1)
    k = logvar_q.shape[-1]

    # -> nB
    term3 = torch.bmm(torch.bmm(mu_q.unsqueeze(1) - mu_p.unsqueeze(1), Sigma_q_inv),
                      mu_q.unsqueeze(-1) - mu_p.unsqueeze(-1)).squeeze()
    term4 = torch.diagonal(torch.bmm(Sigma_q_inv, Sigma_p), dim1=-2, dim2=-1).sum(-1)
    return (1/2) * (term1 - k + term3 + term4)


import torch.nn.functional as F
def log_pdf_mog(z_sample, means, logvars, component_weights=None, st_softmax=False):
    """
    See Eq. 22 of the vampprior paper

    This actually calculates the log pdf of any mixture of gaussians, not just vampprior
    :param self:
    :param z_sample: nB*nO, latent_dim
    :param means:
    :param logvars:
    :return: PDF value of each scalar z according to vampprior.
    """
    # assume nO==1
    # nB*nO, 1, latent_dim
    z_sample = z_sample.unsqueeze(1)

    if len(means.shape) == 3:
        # nB, num_components, 3
        num_components = means.shape[1]
    else:
        # len==2
        num_components = means.shape[0]

    if (means.shape == 2):
        # if using a fixed set of mog components
        # unsqueeze batch dim
        # -> 1 x num_components x latent_dim
        means = means.unsqueeze(0)
        logvars = logvars.unsqueeze(0)

    # make distributions of all the components of the rotated_pc_mean, such that we can retrieve the PDF from these distributions
    # because we have scalar logvariances, the normal has diagonal covariance matrix
    d = torch.distributions.normal.Normal(means.to(z_sample.device),
                                          torch.exp(logvars.to(z_sample.device)) ** (1 / 2))

    # z_sample should broadcast along the 1 dimension,
    # such that we collect a log_prob from every component of the vampprior
    # -> nB, num_components, latent_dim
    print("ln76 z_sample")
    print(z_sample)

    if component_weights is not None:
        if st_softmax:
            w = F.softmax(component_weights.to(z_sample.device), dim=-1)
        else:
            w = F.softmax(component_weights.to(z_sample.device), dim=-1)

        # the log(w) is analogous to the 1/k term in vampprior paper eq 22
        # TODO: is the log(w) calculation here correct...
        if len(w.shape) == 2:
            # d.log_prob: nB, num_components, latent_dim
            log_p = d.log_prob(z_sample)
        else:
            raise NotImplementedError
            log_p = d.log_prob(z_sample) + torch.log(w).unsqueeze(0).unsqueeze(-1)
    else:
        log_p = d.log_prob(z_sample) + math.log(1/num_components)

    # sum over log probabilities, because uncorrelated multivariate normal is same as independent normal
    # nB, num_components, latent_dim -> nB, num_components
    log_p = log_p.sum(-1)

    # sum (nB, num_components) joint prob of q and (nB, num_components) weight vector -> nB, num_components
    log_p = log_p + torch.log(w + 1E-8)

    # logsumexp for MoG, where we sum over the num_components dimension
    log_prob = torch.logsumexp(log_p, dim=1, keepdim=False)

    # -> nB
    return log_prob


def calculate_kld_loss_unreduced(
        kl_type,
        encoder_z_mu=None,
        encoder_z_logvar=None,
        encoder_z_sample=None,
        encoder_cat_probabilities=None,
        prior_z_sample=None,
        prior_mog_z_mu=None,
        prior_mog_z_logvar=None,
        prior_mog_weights=None,
        encoder_flow_log_prob=None,
        prior_flow_log_prob=None,
        **kwargs
):
    # assert (prior_z_sample is None) != (encoder_z_sample is None)
    if encoder_z_sample is None:
        z_sample = prior_z_sample
    else:
        z_sample = encoder_z_sample

    if "standard_gaussian" not in kl_type:
        z_sample = z_sample.to(encoder_z_mu.device)
    if "standard_gaussian" in kl_type:
        kld_loss_unreduced = KL_divergence_gaussians_standard_prior(encoder_z_mu, encoder_z_logvar)
    elif "closed_form_normals" in kl_type:
        kld_loss_unreduced = KL_divergence_independent_multivariate_gaussians(encoder_z_mu,
                                                                              prior_mog_z_mu,
                                                                              encoder_z_logvar,
                                                                              prior_mog_z_logvar)
    elif "categorical" in kl_type:
        kld_loss_unreduced = KL_divergence_categorical(encoder_cat_probabilities)
    elif "vamp" in kl_type or "mog" in kl_type:
        # -> nB
        log_p_z_sample = log_pdf_mog(z_sample, prior_mog_z_mu, prior_mog_z_logvar,
                                     component_weights=prior_mog_weights)

        posterior = torch.distributions.normal.Normal(encoder_z_mu, torch.exp(encoder_z_logvar) ** (1 / 2))

        # nB, latent_dim -> nB
        # multiply independent probabilites by summing their logs
        log_q_z_sample = posterior.log_prob(z_sample).sum(dim=-1)

        # this is a MC estimator of KLD(Q || P)
        # it makes sense since encoder_z_sample comes from the encoder distribution
        # from vampprior
        # nB AND nB -> nB

        assert log_p_z_sample.shape == log_q_z_sample.shape
        assert len(log_p_z_sample.shape) == 1
        if encoder_z_sample is None:
            # prior_z_sample is active
            kld_loss_unreduced = log_p_z_sample - log_q_z_sample
        else:
            # encoder_z_sample is active
            kld_loss_unreduced = log_q_z_sample - log_p_z_sample
    elif "flow_mog" in kl_type:
        log_p_z_sample = log_pdf_mog(z_sample, prior_mog_z_mu, prior_mog_z_logvar,
                                     component_weights=prior_mog_weights)

        # get the logprob from the flow
        import pdb
        pdb.set_trace()
        # TODO i think the sum(-1) is wrong
        kld_loss_unreduced = encoder_flow_log_prob - log_p_z_sample.sum(-1)
    elif "normals_flow" in kl_type:
        posterior = torch.distributions.normal.Normal(encoder_z_mu, torch.exp(encoder_z_logvar) ** (1 / 2))
        log_q_z_sample = posterior.log_prob(z_sample).sum(dim=-1)

        import pdb
        pdb.set_trace()
        # TODO is this unreduced
        kl_divergence = log_q_z_sample - prior_flow_log_prob
    else:
        raise NotImplementedError
    return kld_loss_unreduced


class CVAELoss(nn.Module):
    def __init__(self,
                 recon_loss_type=None,
                 kld_weight=.001,
                 kl_type="gaussian",
                 sigmoid_variance=None,
                 sigmoid_offset=None,
                 sigmoid_flip=False,
                 positive_class_weight=None,
                 iteration_count=0,
                 final_target_capacity=None,
                 final_target_capacity_iteration=None,
                 prior_kld_weight=1,
                 trans_weight=None,
                 kld_lower=-100000,
                 kld_upper=100000,
                 equalize_num_pos_num_neg=True
        ):
        """

        :param recon_loss_type:
        :param kld_weight:
        :param kl_type:
        :param sigmoid_variance:
        :param sigmoid_offset:
        :param sigmoid_flip: whether or not to flip the sigmoid and anneal beta from 1 to 0
        :param positive_class_weight: the higher this is, the more we prioritize background points... because background points are the positive class
        :param iteration_count:
        :param final_target_capacity:
        :param final_target_capacity_iteration:
        :param prior_kld_weight: The scaling factor AFTER first applying the kld weight.
            In other words, this describes the relative weight between the posterior kld and prior kld
        """
        super().__init__()
        self.kld_weight = kld_weight
        self.recon_loss_type = recon_loss_type
        self.iteration_count = iteration_count
        self.sigmoid_variance = sigmoid_variance
        self.sigmoid_offset = sigmoid_offset
        self.sigmoid_flip = sigmoid_flip
        self.kl_type = kl_type
        self.positive_class_weight = positive_class_weight

        self.total_iterations_till_max_capacity = final_target_capacity_iteration

        self.prior_kld_weight = prior_kld_weight
        if "target_capacity" in self.kl_type:
            self.final_target_capacity = final_target_capacity

        if "trans" in self.recon_loss_type:
            self.trans_weight = trans_weight
        self.kld_lower = kld_lower
        self.kld_upper = kld_upper
        self.equalize_num_pos_num_neg = equalize_num_pos_num_neg

    def forward(self,
                predictions,
                truth_labels,
                encoder_z_mu=None,
                encoder_z_logvar=None,
                encoder_z_sample=None,
                encoder_cat_probabilities=None,
                encoder_trans=None,
                prior_mog_z_mu=None,
                prior_mog_z_logvar=None,
                prior_mog_weights=None,
                prior_trans=None,
                prior_z_sample=None,
                encoder_flow_log_prob=None,
                prior_flow_log_prob=None,
                increment_iteration=True,
                decoder_logvar=None,
                decoder_pc=None,
                canonical_pc=None,
                rotated_quat_matrices=None,
                rotated_pc=None,
                kld_lower=-1000,
                kld_upper=1000
                ):
        """

        :param predictions: nB, 4
        :param truth_labels: nB, 4 np.ndarray
        :param encoder_z_mu: nB, 4
        :param encoder_z_logvar: nB, 4
        :param encoder_z_sample: from encoder reparam trick
        :param prior_mog_z_mu: the predicted/fixed parameters of the MOG/VAMP Prior
        :param rotated_quat_matrices: matrices representing the rotation from the CANONICAL to the ROTATED PC
        :return:
        """
        extra_log_dict = dict()

        if increment_iteration:
            self.iteration_count += 1
        if "chordal" in self.recon_loss_type:
            reconstruction = chordal_loss(torch.Tensor(truth_labels).to(predictions.device).unsqueeze(1),
                                          predictions.unsqueeze(1))
        elif "pc2pc" in self.recon_loss_type:
            # pointcloud to pointcloud
            reconstruction = torch.mean(torch.norm(predictions - truth_labels.squeeze(1).to(predictions.device), dim=-1))
        elif "rot_frobenius" in self.recon_loss_type:
            # reconstruction = torch.mean((1/2) * torch.linalg.norm(predictions - torch.Tensor(truth_labels).to(predictions.device),
            #                                    dim=[-1, -2])**2)
            reconstruction = torch.mean((1/2) * torch.linalg.norm(predictions -
                                                                  torch.Tensor(np.linalg.inv(rotated_quat_matrices)).to(predictions.device),
                                               dim=[-1, -2])**2)
        elif "rot_frobenius_pc_regularized" in self.recon_loss_type:
            # rot_quat_matrix: rotation from canonical to rotated pc
            # therefore, the inverse brings us from rot pc to canonical
            rot_loss = torch.mean((1/2) * torch.linalg.norm(predictions -
                                                            torch.Tensor(R.from_matrix(rotated_quat_matrices).inv().as_matrix()).to(predictions.device),
                                                            dim=[-1, -2])**2)

            # relative_per_point_translations: takes us from rot pc to canonical pc
            # B - A
            relative_per_point_translations = canonical_pc.squeeze(1).squeeze(1) - rotated_pc.squeeze(1)
            point_match_loss = torch.mean(torch.norm(decoder_pc.transpose(-1, -2) -
                                                     relative_per_point_translations.to(predictions.device), dim=-1))
            extra_log_dict['rot_loss'] = rot_loss
            extra_log_dict['point_match_loss'] = point_match_loss
            reconstruction = rot_loss + point_match_loss
        elif "rot_frobenius_calibrated" in self.recon_loss_type:
            channel_dim = 9
            sum_squared_errors = (1/2) * torch.linalg.norm(predictions - torch.Tensor(truth_labels).to(predictions.device),
                                                           dim=[-1, -2])**2

            # nB, 1 -> nB
            decoder_logvar = decoder_logvar.squeeze(-1)
            # both should be nB
            assert sum_squared_errors.shape == decoder_logvar.shape

            reconstruction = torch.mean(1/(2*torch.exp(decoder_logvar))*sum_squared_errors +
                                        channel_dim*torch.log(torch.exp(decoder_logvar)**(1/2)))
        else:
            # bce
            # truth_labels: nB, num_points
            if self.equalize_num_pos_num_neg:
                assert torch.sum(truth_labels) > 0
                # positive aka background labels
                num_points = truth_labels.shape[-1]

                assert torch.sum(truth_labels) < num_points * truth_labels.shape[0]

                positive_label_count = torch.sum(truth_labels, axis=-1)

                negative_label_count = num_points - positive_label_count
                np_ratio = negative_label_count / positive_label_count

                pos_weight = truth_labels * np_ratio.unsqueeze(-1) * self.positive_class_weight + (1-truth_labels)
                loss_fnx = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(predictions.device))
            elif self.positive_class_weight is not None:
                # scale each positive label loss according to self.positive_class_weight
                assert torch.sum(truth_labels) > 0
                pos_weight = truth_labels * self.positive_class_weight + torch.ones_like(truth_labels)
                loss_fnx = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(predictions.device))
            else:
                loss_fnx = nn.BCEWithLogitsLoss()

            assert len(predictions.shape) == 2 == len(truth_labels.shape), (predictions.shape, truth_labels.shape)
            reconstruction = loss_fnx(predictions, truth_labels.to(predictions.device))

        if self.sigmoid_variance is not None:
            assert len(self.kld_weight) == 2, self.kld_weight
            kl_start = self.kld_weight[0]
            kl_end = self.kld_weight[1]

            sigmoid_input = torch.tensor([self.iteration_count/self.sigmoid_variance], requires_grad=False).to(predictions.device)\
                            - self.sigmoid_offset

            if self.sigmoid_flip:
                kld_weight = kl_start * (1-torch.sigmoid(sigmoid_input)) + kl_end
            else:
                kld_weight = kl_end * torch.sigmoid(sigmoid_input) + kl_start

        else:
            kld_weight = self.kld_weight

        print("ln422 kld_weight")
        print(kld_weight)
        self.kld_lower = kld_lower
        self.kld_upper = kld_upper

        posterior_kld_loss_unreduced = calculate_kld_loss_unreduced(
            self.kl_type,
            encoder_z_mu=encoder_z_mu,
            encoder_z_logvar=encoder_z_logvar,
            encoder_z_sample=encoder_z_sample,
            encoder_cat_probabilities=encoder_cat_probabilities,
            prior_mog_z_mu=prior_mog_z_mu,
            prior_mog_z_logvar=prior_mog_z_logvar,
            prior_mog_weights=prior_mog_weights,
            encoder_flow_log_prob=encoder_flow_log_prob,
            prior_flow_log_prob=prior_flow_log_prob,
        )
        extra_log_dict['posterior_kld_loss_unreduced'] = posterior_kld_loss_unreduced.mean().squeeze()

        if "target_capacity" in self.kl_type:
            import pdb
            pdb.set_trace()
            # TODO unreduced
            cc = calculate_current_capacity(self.iteration_count,
                                            self.final_target_capacity,
                                            self.total_iterations_till_max_capacity)
            kld_loss = torch.abs(kld_loss - cc)
            extra_log_dict['current_capacity'] = cc

        total_loss = reconstruction + kld_weight * \
                     torch.clamp(posterior_kld_loss_unreduced, self.kld_lower, self.kld_upper).mean().to(reconstruction.device)

        if "encoder_trans" in self.recon_loss_type:
            # encourage encoder trans to be orthogonal
            total_loss += feature_transform_regularizer(encoder_trans).to(total_loss.device)
        if "prior_trans" in self.recon_loss_type:
            total_loss += feature_transform_regularizer(prior_trans).to(total_loss.device)

        prior_kld_loss_unreduced = calculate_kld_loss_unreduced(
            self.kl_type,
            encoder_z_mu=encoder_z_mu,
            encoder_z_logvar=encoder_z_logvar,
            prior_z_sample=prior_z_sample,
            encoder_cat_probabilities=encoder_cat_probabilities,
            prior_mog_z_mu=prior_mog_z_mu,
            prior_mog_z_logvar=prior_mog_z_logvar,
            prior_mog_weights=prior_mog_weights,
            encoder_flow_log_prob=encoder_flow_log_prob,
            prior_flow_log_prob=prior_flow_log_prob,
        )

        extra_log_dict['prior_kld_loss_unreduced'] = prior_kld_loss_unreduced.mean().squeeze()

        if "prior_kld" in self.kl_type:
            total_loss += torch.clamp(prior_kld_loss_unreduced, self.kld_lower, self.kld_upper).mean().to(total_loss.device) \
                          * self.prior_kld_weight * kld_weight

        return total_loss, reconstruction, posterior_kld_loss_unreduced.mean(), kld_weight, extra_log_dict
