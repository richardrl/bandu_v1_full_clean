import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        # -> nB, 1024
        x = x.view(-1, 1024)

        try:
            x = F.relu(self.bn4(self.fc1(x)))
        except:
            import pdb
            pdb.set_trace()
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, stn_transform=False,
                 feature_transform=False, channel=3, batchnorm=False, max_dim=1024):
        super(PointNetEncoder, self).__init__()
        # self.stn = STN3d(3)

        # self.stn = STN3d(channel)

        self.conv1 = torch.nn.Conv1d(channel, 64, 1) # 1x1 convolution -> fcl
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, max_dim, 1)
        self.max_dim = max_dim
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(self.max_dim)
        self.global_feat = global_feat
        self.stn_transform = stn_transform
        if self.stn_transform:
            self.stn = STN3d(channel)

        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.use_batchnorm = batchnorm

    def forward(self, x, ret_global_feat=False):
        assert isinstance(x, torch.Tensor)
        # nB * nO, feature_dim == 4, num_points
        B, D, N = x.size()

        """
        Joint Alignment Network begin
        spatial transformer
        
        only transforms first 3 XYZ channels
        """

        # transformation, AKA rotation AKA orthogonal matrix
        if self.stn_transform:
            # trans = self.stn(x[:, :3, :])
            trans = self.stn(x)

            # -> B, N, D
            x = x.transpose(2, 1)

            if D > 3:
                feature = x[:, :, 3:]
            x = x[:, :, :3]
            x = torch.bmm(x, trans)

            if D > 3:
                x = torch.cat([x, feature], dim=2)

            # -> B, D, N
            x = x.transpose(2, 1)

        # conv1 takes in B, D (channels), N (length)
        # -> nB * nO, 64, num_points

        """
        Joint Alignment Network end
        """

        # if this line fails: did you run .permute(0, 2, 1) on the input x?
        if self.use_batchnorm:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.conv1(x))
        #
        # if self.feature_transform:
        #     # this does the canonicalization thing
        #     trans_feat = self.fstn(x)
        #     x = x.transpose(2, 1)
        #     x = torch.bmm(x, trans_feat)
        #     x = x.transpose(2, 1)
        # else:
        #     trans_feat = None

        # B, 128, N: features of individual points
        pointfeat = x

        # -> B, 128, N
        if self.use_batchnorm:
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = F.relu(self.conv2(x))

        # -> B, 1024, N
        if self.use_batchnorm:
            x = self.bn3(self.conv3(x))
        else:
            x = self.conv3(x)

        # Pool over the points dimension
        # nB * nO, 1024, num_points -> nB * nO, 1024, 1
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.max_dim)

        if self.stn_transform:
            return x, torch.cat([x.unsqueeze(-1).expand(-1, -1, N), pointfeat], 1), trans
        else:
            return x, torch.cat([x.unsqueeze(-1).expand(-1, -1, N), pointfeat], 1)

        #     x = x.view(-1, 1024, 1).repeat(1, 1, N)
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss