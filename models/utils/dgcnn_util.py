import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None, device=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph

            # nB, num_points, num_neighbors
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    if device is None:
        device = torch.device('cuda')

    # -> nB, 1, 1
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    # -> nB, num_points, num_neighbors
    idx = idx + idx_base.to(idx.device)

    # -> nB * num_points * num_neighbors
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)

    # -> (batch_size*num_points, num_dims)
    #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]

    # -> nB, num_points, num_neighbors, channels
    feature = feature.view(batch_size, num_points, k, num_dims)

    # nB, num_points, 1, channels -> nB, num_points, num_neighbors, channels
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # the concatenation is basically doing the dot product of self attention
    # -> nB, num_points, num_neighbors, num_dims * 2
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature