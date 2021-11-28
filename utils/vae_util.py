from torch import nn
import torch
import tqdm
from torch.nn import functional as F
from utils import loss_util


def marginalize_likelihood_binary(model,
                                  batch,
                                  total_num_z_per_sample,
                                  per_batch_num_z=1):
    """
    
    :param model: 
    :param batch: 
    :param total_num_z_per_sample: 
    :return: 
    """
    # sample_idx a bunch of z^(l) samples from the prior p(z|x)
    # feed z into the decoder p(y|x,z^(l)) to get binary logits
    # the log likelihood of binary logits is the binary cross entropy loss
    # we average over the BCE loss for a fixed y, inserted into many p(y|x,z^(l))
    # the bce tells us, given a distribution parameterized by the predicted logits, how likely is our
    # one-hot contact mask
    loss_fnx = nn.BCEWithLogitsLoss(reduction="none")

    # nB, num_points
    target = batch['bottom_thresholded_boolean'].squeeze(1).squeeze(-1)

    nB, num_points = target.shape

    log_probabilities = []
    with torch.no_grad():
        for batch_idx in tqdm.tqdm(range(total_num_z_per_sample // per_batch_num_z)):
            model_reconstructions = model.decode_batch(batch,
                                                       z_samples_per_sample=per_batch_num_z)

            nB, nZ, num_points = model_reconstructions.shape

            # unreduced loss: -> nB, num_z_samples, num_points
            #   this loss is the log probabilities
            target_reshaped = target.unsqueeze(1).expand(-1, per_batch_num_z, -1).to(model_reconstructions.device)

            # combine first two dimensions of each
            likelihood = loss_fnx(model_reconstructions.reshape(nB*nZ, num_points),
                                  target_reshaped.reshape(nB*nZ, num_points))

            # reduce over the last dimension because the per-point distributions are assumed independent
            # -> nB, num_z_samples
            joint_likelihood = likelihood.sum(-1)

            # logsumexp over the z_sample dimension because we need to average the densities over the z_sample dimension
            # -> nB
            loss = torch.logsumexp(joint_likelihood, dim=-1)

            log_probabilities.append(loss)
        # -> total_batches, per_batch_num_z
        stacked = torch.stack(log_probabilities).T

        # sum over each small batch, then sum over the entire main "batch"
        # add normalization factor
        stacked = stacked + torch.log(torch.Tensor([1/total_num_z_per_sample]).to(stacked.device))
        final_log_likelihoods = torch.logsumexp(stacked, dim=-1)
        print("final log likelihoods")
        print(final_log_likelihoods)

        final_log_likelihoods = final_log_likelihoods + torch.log(torch.Tensor([1/nB])).to(stacked.device)
        average = torch.logsumexp(final_log_likelihoods, dim=-1)
        print('average')
        print(average)

    return final_log_likelihoods


def forward_kl(model,
               batch,
               kld_type
               ):
    """
    Forward KL using samples from the prior
    :param model:
    :param batch:
    :param kld_type:
    :return:
    """
    # sample_idx from prior
    # feed these samples into both the posterior and the prior to get the likelihoods
    # the average of the log differences is the estimated KL divergence

    # -> nB, num_z_samples = 1, 1, 128 -> nB, 128
    prior_sampled_z = model.sample_prior(batch, z_samples_per_sample=1).squeeze()

    # calculate encoder and decoder distribution parameters using model forward pass
    forward_dict = model.forward(batch)


    # plugin z_samples from decoder as the ENCODER_Z_SAMPLE into encoder and decoder distribution

    del forward_dict['encoder']['encoder_z_sample']
    # import pdb
    # pdb.set_trace()
    kld_loss_unreduced = loss_util.calculate_kld_loss_unreduced(kld_type,
                                                                # prior_z_sample=prior_sampled_z,
                                           **forward_dict['encoder'],
                                           **forward_dict['prior'])
    return kld_loss_unreduced


def reverse_kl(model,
               batch,
               kld_type
               ):
    """
    Reverse KL using samples from the posterior
    :param model:
    :param batch:
    :param kld_type:
    :return:
    """
    # sample_idx from prior
    # feed these samples into both the posterior and the prior to get the likelihoods
    # the average of the log differences is the estimated KL divergence

    # calculate encoder and decoder distribution parameters using model forward pass
    forward_dict = model.forward(batch)


    # plugin z_samples from decoder as the ENCODER_Z_SAMPLE into encoder and decoder distribution
    del forward_dict['prior']['prior_z_sample']
    kld_loss_unreduced = loss_util.calculate_kld_loss_unreduced(kld_type,
                                                                **forward_dict['encoder'],
                                                                **forward_dict['prior'])
    return kld_loss_unreduced




def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    softmaxed_y, y_presoftmax_logits = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return softmaxed_y.view(-1, latent_dim * categorical_dim), y_presoftmax_logits

    # straight through
    shape = softmaxed_y.size()
    _, ind = softmaxed_y.max(dim=-1)
    y_hard = torch.zeros_like(softmaxed_y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - softmaxed_y).detach() + softmaxed_y
    return y_hard.view(-1, latent_dim * categorical_dim), y_presoftmax_logits


def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape)
    if device:
        U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y_presoftmax_logits = logits + sample_gumbel(logits.size(), device=logits.device).to(logits.device)
    return F.softmax(y_presoftmax_logits / temperature, dim=-1), y_presoftmax_logits