import torch.nn as nn
import torch
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim, batchnorm=False, activation_fn="leaky_relu"):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)

        if activation_fn == "leaky_relu":
            self.activation_fn = F.leaky_relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.tanh
        else:
            raise NotImplementedError
        # self.bn = nn.BatchNorm3d

    def forward(self, vertices):
        return vertices + self.activation_fn(self.linear(vertices))