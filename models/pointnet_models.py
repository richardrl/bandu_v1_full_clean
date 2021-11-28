import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn

from utils.pointnet_utils import PointNetEncoder


class PointnetCls(nn.Module):
    def __init__(self, output_dim=40,
                 channel=3,
                 normal_channel=True,
                 stn_transform=False,
                 feature_transform=False,
                 dropout=False,
                 batchnorm=False,
                 max_dim=1024):
        super(PointnetCls, self).__init__()
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        self.feat1 = PointNetEncoder(batchnorm=batchnorm,
                                     global_feat=False,
                                     stn_transform=stn_transform,
                                     feature_transform=feature_transform, channel=channel,
                                     max_dim=max_dim)

        self.feat2 = PointNetEncoder(batchnorm=batchnorm,
                                     global_feat=False,
                                     feature_transform=feature_transform,
                                     stn_transform=False,
                                     channel=max_dim + 64,
                                     max_dim=max_dim)

        # this ones takes in point dim + global feature from before
        self.feat3 = PointNetEncoder(batchnorm=batchnorm,
                                     stn_transform=False,
                                     global_feat=True,
                                     feature_transform=feature_transform,
                                     channel=max_dim + 64,
                                     max_dim=max_dim)
        self.max_dim = max_dim
        self.fc1 = nn.Linear(max_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.use_dropout = dropout
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.use_batchnorm = batchnorm

    def forward(self, x, ret_global_feat=False):
        # x, trans, trans_feat = self.feat(x)
        # -> nB, 1024 + 64, num_points

        if self.feat1.stn_transform:
            global_feat, concatenated_points, trans = self.feat1(x)
        else:
            global_feat, concatenated_points = self.feat1(x)

        # -> nB, 1024 + 64, num_points
        global_feat, concatenated_points = self.feat2(concatenated_points)

        # -> nB, 1024
        # x: a global pooled feature
        global_feat, concatenated_points = self.feat3(concatenated_points)

        if self.use_batchnorm:
            # bn expects channel
            x = F.relu(self.bn1(self.fc1(global_feat)))
        else:
            x = F.relu(self.fc1(global_feat))

        if self.use_dropout:
            assert self.use_batchnorm
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            if self.use_batchnorm:
                x = F.relu(self.bn2(self.fc2(x)))
            else:
                x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        # return x, trans_feat
        # -> nB * nO, embedding_dim

        # if self.feat1.stn_transform:
        #     return x, global_feat, trans
        # else:
        #     return x, global_feat
        # else:
        #     if self.feat1.stn_transform:
        #         return x, trans
        #     else:
        #         return x
        if self.feat1.stn_transform:
            return x, trans
        else:
            return x

class PointnetSeg(nn.Module):
    def __init__(self,
                 output_dim=1,
                 channel=3,
                 stn_transform=False,
                 normal_channel=True,
                 feature_transform=False,
                 dropout=False,
                 dropout_prob=.4,
                 batchnorm=False,
                 max_dim=1024,
                 skip_encoders=False):
        """

        :param output_dim:
        :param channel:
        :param normal_channel:
        :param feature_transform:
        :param dropout:
        :param dropout_prob:
        :param batchnorm:
        :param max_dim:
        :param skip_encoders: Whether or not to cripple the decoder
        """
        super(PointnetSeg, self).__init__()
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3

        self.skip_encoders = skip_encoders
        if skip_encoders:
            self.fc1 = nn.Linear(channel, 512)
        else:
            self.feat1 = PointNetEncoder(batchnorm=batchnorm, global_feat=False,
                                         stn_transform=stn_transform,
                                         feature_transform=feature_transform,
                                         channel=channel, max_dim=max_dim)

            self.feat2 = PointNetEncoder(batchnorm=batchnorm, global_feat=False,
                                         stn_transform=False,
                                         feature_transform=feature_transform,
                                         channel=max_dim + 64, max_dim=max_dim)

            # this ones takes in point dim + global feature from before
            self.feat3 = PointNetEncoder(batchnorm=batchnorm, global_feat=False,
                                         stn_transform=False,
                                         feature_transform=feature_transform,
                                         channel=max_dim + 64, max_dim=max_dim)

            self.fc1 = nn.Linear(max_dim + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout
        self.output_dim = output_dim

        if batchnorm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.use_batchnorm = batchnorm

    def forward(self, x):
        if self.skip_encoders:
            pass
        else:
            # x, trans, trans_feat = self.feat(x)
            # -> nB, 1024 + 64, num_points

            if self.feat1.stn_transform:
                global_feat, x, trans = self.feat1(x)
            else:
                global_feat, x = self.feat1(x)

            # -> nB, 1024 + 64, num_points
            global_feat, x = self.feat2(x)

            # -> nB, 1024 + 64, num_points
            global_feat, x = self.feat3(x)

        # -> nB, num_points, 1088
        x = x.transpose(-1, -2)
        if self.use_batchnorm:
            # -> nB, num_points, 512
            x = self.fc1(x)

            # -> nB, 512, num_points
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = F.relu(self.bn1(x))

            # -> nB, num_points, 512
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)
        else:
            x = F.relu(self.fc1(x))


        if self.use_batchnorm:
            # -> nB, num_points, 256
            x = self.fc2(x)

            # -> nB, 256, num_points
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = self.bn2(x)

            # -> nB, num_points, 256
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = F.relu(x)

            if self.use_dropout:
                x = self.dropout(x)
        else:
            assert not self.use_dropout, "Dropout always with BN for now"
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        # return x, trans_feat
        # -> nB * nO, embedding_dim

        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        if not self.skip_encoders and self.feat1.stn_transform:
            return x, trans
        else:
            return x

class PointnetSegAndCls(nn.Module):
    def __init__(self,
                 seg_output_dim=1,
                 cls_output_dim=5,
                 channel=3,
                 normal_channel=True,
                 stn_transform=False,
                 feature_transform=False,
                 dropout=False,
                 dropout_prob=.4,
                 batchnorm=False,
                 max_dim=1024):
        """
        combines seg and cls models into one
        :param output_dim:
        :param channel:
        :param normal_channel:
        :param feature_transform:
        :param dropout:
        :param dropout_prob:
        :param batchnorm:
        :param max_dim:
        """
        super(PointnetSegAndCls, self).__init__()
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        self.feat1 = PointNetEncoder(batchnorm=batchnorm, global_feat=False,
                                     stn_transform=stn_transform,
                                     feature_transform=feature_transform,
                                     channel=channel, max_dim=max_dim)

        self.feat2 = PointNetEncoder(batchnorm=batchnorm, global_feat=False, feature_transform=feature_transform,
                                     channel=max_dim + 64, max_dim=max_dim)

        # this ones takes in point dim + global feature from before
        self.feat3 = PointNetEncoder(batchnorm=batchnorm, global_feat=False, feature_transform=feature_transform,
                                     channel=max_dim + 64, max_dim=max_dim)

        self.fc1 = nn.Linear(max_dim + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, seg_output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_dropout = dropout

        if batchnorm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

        self.stn_transform = stn_transform

        self.relu = nn.ReLU()
        self.use_batchnorm = batchnorm

        # cls part
        self.fc1_cl = nn.Linear(max_dim, 512)
        self.fc2_cl = nn.Linear(512, 256)
        self.fc3_cl = nn.Linear(256, cls_output_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.use_dropout = dropout
        self.bn1_cl = nn.BatchNorm1d(512)
        self.bn2_cl = nn.BatchNorm1d(256)

    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)
        # -> nB, 1024 + 64, num_points

        if self.stn_transform:
            global_feat, concatenated_pts, trans = self.feat1(x)
        else:
            global_feat, concatenated_pts = self.feat1(x)

        # -> nB, 1024 + 64, num_points
        global_feat, concatenated_pts = self.feat2(concatenated_pts)

        # -> nB, 1024 + 64, num_points
        global_feat, x = self.feat3(concatenated_pts, ret_global_feat=True)

        # -> nB, num_points, 1088

        # Make pseudo inputs
        # x = x.permute(0, 2, 1)
        x = x.transpose(-1, -2)
        if self.use_batchnorm:
            # -> nB, num_points, 512
            x = self.fc1(x)

            # -> nB, 512, num_points
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = F.relu(self.bn1(x))

            # -> nB, num_points, 512
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)
        else:
            x = F.relu(self.fc1(x))


        if self.use_batchnorm:
            # -> nB, num_points, 256
            x = self.fc2(x)

            # -> nB, 256, num_points
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = self.bn2(x)

            # -> nB, num_points, 256
            # x = x.permute(0, 2, 1)
            x = x.transpose(-1, -2)

            x = F.relu(x)

            if self.use_dropout:
                x = self.dropout(x)
        else:
            assert not self.use_dropout, "Dropout always with BN for now"
            x = F.relu(self.fc2(x))
        pseudo_inputs = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        # return x, trans_feat
        # -> nB * nO, embedding_dim

        # make categorical vector
        if self.use_batchnorm:
            # bn expects channel
            x = F.relu(self.bn1_cl(self.fc1_cl(global_feat)))
        else:
            x = F.relu(self.fc1_cl(global_feat))

        if self.use_dropout:
            assert self.use_batchnorm
            x = F.relu(self.bn2_cl(self.dropout(self.fc2_cl(x))))
        else:
            if self.use_batchnorm:
                x = F.relu(self.bn2_cl(self.fc2_cl(x)))
            else:
                x = F.relu(self.fc2_cl(x))

        categorical_logits = self.fc3_cl(x)

        pseudo_inputs = pseudo_inputs.unsqueeze(-1).transpose(1, 2)

        # nB, num_mog_prior_components, num_points, 1 AND nB, num_mog_prior_components
        if self.stn_transform:
            return pseudo_inputs, categorical_logits, trans
        else:
            return pseudo_inputs, categorical_logits


class PointnetClsAndCls(nn.Module):
    def __init__(self, output1_dim, output2_dim,
                 channel1,
                 channel2):
        super().__init__()
        self.cls1 = PointnetCls(output_dim=output1_dim,
                                channel=channel1)
        self.cls2 = PointnetCls(output_dim=output2_dim,
                                channel=channel2)

    def forward(self, batch):
        out1 = self.cls1(batch)


class PointNetQuaternion(PointnetCls):
    def __init__(self, normalize=True, *args, **kwargs):
        kwargs['output_dim'] = 4
        super().__init__(*args, **kwargs)
        self.normalize = normalize

    def forward(self, batch):
        # nB, nO==1, num_points, 3 -> nB, 4
        nB, nO, num_points, _ = batch['rotated_pointcloud'].shape

        pc = batch['rotated_pointcloud']

        if self.normalize:
            pc = (pc - batch['dataset_mean'])/(torch.sqrt(batch['dataset_var']) + 1E-6)
            assert torch.sum(torch.isnan(pc)) == 0

        if nB > 1:
            assert not torch.all(batch['rotated_pointcloud'] == batch['rotated_pointcloud'][0][0])

        quat = super().forward(pc.squeeze(1).permute(0, 2, 1))
        return quat / torch.norm(quat, dim=-1, keepdim=True)


class PointNetSegBinary(PointnetSeg):
    def __init__(self, *args, **kwargs):
        kwargs['output_dim'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        # nB, nO==1, num_points, 3 -> nB, 4
        nB, nO, num_points, _ = batch['rotated_pointcloud'].shape

        # collapse nB and nO dimensions
        binary_logits = super().forward(batch['rotated_pointcloud'].reshape(-1, num_points, _).permute(0, 2, 1))

        # -> nB, num_points
        return binary_logits.squeeze(-1)


