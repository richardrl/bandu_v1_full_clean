from nflows import transforms, distributions, flows
from nflows.flows.base import Flow
from bandu.torch.modules import ResidualBlock
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
# from nflows.nn.nets.resnet import ResidualBlock

import torch
import torch.nn as nn
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from nflows.transforms.normalization import BatchNorm


class ContextualResidualBlock(ResidualBlock):
    def __init__(self, embedding_dim, batchnorm=True):
        super().__init__(embedding_dim, batchnorm=batchnorm)

    def forward(self, inputs, context=None):
        return super().forward(inputs)


class ContextualMlp(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, batchnorm_within_layer, num_hidden_layers_within_layer):
        super().__init__()
        self.il = nn.Linear(input_dim, embedding_dim)
        self.ol = nn.Linear(embedding_dim, output_dim)
        self.hl = [ContextualResidualBlock(embedding_dim,
                            batchnorm=batchnorm_within_layer) for _ in range(num_hidden_layers_within_layer)]
        # self.hl = [ResidualBlock() for _ in range(num_hidden_layers_within_layer)]

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.il(inputs)
        else:
            temps = self.il(torch.cat((inputs, context), dim=1))
        for block in self.hl:
            temps = block(temps, context=context)
        outputs = self.ol(temps)
        return outputs

class LatentRealNVP(Flow):
    def __init__(self,
                 input_features,
                 embedding_dim,
                 num_layers,
                 batchnorm_within_layer=True,
                 num_hidden_layers_within_layer=2,
                 batchnorm_between_layers=True):
        def create_contextual_mlp(input_dim, output_dim):
            return ContextualMlp(input_dim,
                                 output_dim,
                                 embedding_dim,
                                 batchnorm_within_layer,
                                 num_hidden_layers_within_layer)

        layers = []
        mask = torch.ones(input_features)

        # set every other latent variable to be masked away/ignored
        mask[::2] = -1
        for l in range(num_layers):
            transform = AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_contextual_mlp
            )
            layers.append(transform)

            mask *= -1
            if batchnorm_between_layers:
                layers.append(BatchNorm(features=input_features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([input_features])
        )