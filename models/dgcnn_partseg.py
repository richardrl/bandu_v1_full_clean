import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .transform_net import Transform_Net
from .utils.dgcnn_util import get_graph_feature


class DGCNNPartSeg(nn.Module):
    def __init__(self,
                 n_knn=20,
                 num_part=50,
                 num_channels=3,
                 normal_channel=False,
                 use_dropout=True,
                 **kwargs):
        super().__init__()
        # self.args = args
        self.n_knn = n_knn
        # self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(num_channels * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)

        self.use_dropout = use_dropout
        self.label_type = "btb"
        if use_dropout:
            self.dp1 = nn.Dropout(p=0.5)
            self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x, l=None):
        batch_size = x.size(0)
        num_points = x.size(2)

        model_device = next(self.parameters()).device

        # because dataloader makes it a double
        x = x.float()
        # nB, num_channels, num_points -> nB, num_channels * 2, num_points, num_neighbors
        # x0 = get_graph_feature(x, k=self.n_knn)

        # t = self.transform_net(x0)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, t)
        # x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.n_knn, device=model_device)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn, device=model_device)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn, device=model_device)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        # l = l.view(batch_size, -1, 1)
        # l = self.conv7(l)

        # x = torch.cat((x, l), dim=1)

        # -> nB, 1024, num_points
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv8(x)

        if self.use_dropout:
            x = self.dp1(x)

        x = self.conv9(x)

        if self.use_dropout:
            x = self.dp2(x)

        x = self.conv10(x)

        x = self.conv11(x)

        trans_feat = None
        # return x, trans_feat
        return x

# class get_loss(torch.nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#
#     def forward(self, pred, target, trans_feat, smoothing=True):
#         ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#         target = target.contiguous().view(-1)
#
#         if smoothing:
#             eps = 0.2
#             n_class = pred.size(1)
#
#             one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
#             one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#             log_prb = F.log_softmax(pred, dim=1)
#
#             loss = -(one_hot * log_prb).sum(dim=1).mean()
#         else:
#             loss = F.cross_entropy(pred, gold, reduction='mean')
#
#         return loss


class DGCNNPartSegBatchWrapper(DGCNNPartSeg):
    """
    Unwraps the batch
    """
    def __init__(self, *args, A_vec_to_quat_head=False, quat_to_rot=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.A_vec_to_quat_head = A_vec_to_quat_head
        self.quat_to_rot = quat_to_rot

    def forward(self, batch):
        model_device = next(self.parameters()).device

        pc_X = batch['rotated_pointcloud'].squeeze(1).to(model_device).float()

        if len(pc_X.shape) == 4:
            pc_X = pc_X.reshape(-1, pc_X.shape[-2], pc_X.shape[-1])

        pc_X = (pc_X - batch['rotated_pointcloud_mean'].to(model_device).float())/(torch.sqrt(batch['rotated_pointcloud_var'].to(model_device).float()) + 1E-6)

        pc_X = pc_X.permute(0, 2, 1)

        seg_logits = super().forward(pc_X)
        return seg_logits