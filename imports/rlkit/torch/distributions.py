import torch
from torch.distributions import Distribution, Normal
import bandu.imports.rlkit.torch.pytorch_util as ptu
from torch.distributions import Categorical, OneHotCategorical, kl_divergence
from torch.distributions import Normal as TorchNormal
from torch.distributions import Beta as TorchBeta
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Independent
from torch.distributions.utils import _sum_rightmost
from _collections import OrderedDict


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                ptu.zeros(self.normal_mean.size(), torch_device=ptu.get_device()),
                ptu.ones(self.normal_std.size(), torch_device=ptu.get_device())
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


# class Independent(Distribution, TorchIndependent):
#     def get_diagnostics(self):
#         return self.base_dist.get_diagnostics()


class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        # stats.update(create_stats_ordered_dict(
        #     'mean',
        #     ptu.get_numpy(self.mean),
        #     # exclude_max_min=True,
        # ))
        # stats.update(create_stats_ordered_dict(
        #     'std',
        #     ptu.get_numpy(self.distribution.stddev),
        # ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


class GaussianMixture(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]

        log_p = torch.stack(log_p, -1)

        # [batch_size, action_dim, num_gaussians] -> [batch_size, num_gaussians]
        # log_p = log_p.sum(dim=1)

        log_weights = torch.log(self.weights[:, :, 0])

        # -> [batch_size, num_gaussions]
        lp = log_weights + log_p


        m = lp.max(dim=1)[0]  # log-sum-exp numerical stability trick
        # log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1))

        log_p_mixture = m + torch.log(torch.exp(lp - m[:, None]).sum(dim=1))

        return log_p_mixture

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        c = ptu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)
