from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    """
    This predicts a 3D transformation matrix
    """
    def __init__(self, point_dim=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(point_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

    def forward(self, x, mask=None):
        # x: [batch_size, feature_dim == 3, num_points]
        batchsize = x.size()[0]

        # -> [batch_size, embedding_dim == 64, num_points]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # -> [batch_size, embedding_dim == 128, num_points]
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # -> [batch_size, embedding_dim == 1024, num_points]
        x = F.relu(self.bn3(self.conv3(x)))


        if mask is not None:
            x = x + ((1-mask) * -99999).unsqueeze(1)

        # -> [batch_size, embedding_dim == 1024, 1]
        x = torch.max(x, 2, keepdim=True)[0]


        # -> [batch_size, embedding_dim == 1024]
        x = x.view(batchsize, 1024)

        # -> [batch_size, embedding_dim == 512]
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)


        # -> [batch_size, embedding_dim == 256]
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)

        # -> [batch_size, embedding_dim == 9]
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
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
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, mask=None):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        raise Exception
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = False,
                 feature_transform = False,
                 mini_transform=True,
                 point_dim=3,
                 output_dim=1024):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(point_dim=point_dim)
        self.output_dim = output_dim
        self.conv1 = torch.nn.Conv1d(point_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.output_dim, 1)
        # self.conv3 = torch.nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.output_dim)
        # self.bn3 = nn.BatchNorm1d(64)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        num_points = 8
        self.mini_transform = mini_transform

    def forward(self, x, mask=None):
        """

        :param x: batch_size X feature_dim == 64, x num_points
        :param mask:
        :return:
        """
        # x: [32, 3, 2500] == [batch_size, feature_dim, num_points]
        batch_size, feature_dim, num_points = x.size()

        # transformation_matrix: [batch_size, 3, 3]
        if self.mini_transform:
            transformation_matrix = self.stn(x, mask=mask)

            # -> [batch_size, feature_dim == 3, num_points
            x = x.transpose(2, 1)

            # [batch_size, num_points, 3] X [batch_size, 3, 3] -> [batch_size, num_points, 3]
            x = torch.bmm(x, transformation_matrix)

            x = x.transpose(2, 1)
        else:
            transformation_matrix = None

        # -> [batch_size, 64, num_points]
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.conv1(x))

        # SKIP by default
        if self.feature_transform:
            trans_feat = self.fstn(x, mask=mask)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        # pointnetfeat: [batch_size, 64, num_points]
        pointfeat = x

        # -> [batch_size, 128, num_points]
        x = F.relu(self.bn2(self.conv2(x)))

        # -> [batch_size, self.output_dim, num_points]
        x = self.bn3(self.conv3(x))

        # [batch_size, self.output_dim, num_points] -> [batch_size, self.output_dim, 1]
        x = torch.max(x, 2, keepdim=True)[0]

        # -> [batch_size, self.output_dim]
        x = x.view(batch_size, self.output_dim)
        # x = x.view(batch_size, 64)

        if self.global_feat:
            return x, transformation_matrix, trans_feat
        else:
            # Concatenate the original pointcloud points to the global feature x
            # -> [batch_size, self.output_dim, num_points]
            x = x.view(batch_size, self.output_dim, 1).repeat(1, 1, num_points)
            # x = x.view(batch_size, 64, 1).repeat(1, 1, num_points)

            # [batch_size, 64 + self.output_dim, num_points], ... , ...
            return torch.cat([x, pointfeat], 1), transformation_matrix, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x.shape == [batch_size, feature_dim, num_points]
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        # -> (batch_size, 1088, num_points), (batch_size, 3, 3), (None)
        x, trans, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())