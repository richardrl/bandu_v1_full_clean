from torch import nn
from bandu.torch.modules import ResidualBlock
import torch
import copy

class MLP(nn.Module):
    def __init__(self, embedding_dim=256, input_type="pb_id", num_objects=2, normalization_type="static", batchnorm=False):
        """

        :param embedding_dim:
        :param input_type:
        :param num_objects:
        :param normalization_type:
        :param rotated_pc_mean: To be used for static normalization
        :param rotated_pc_var: To be used for static normalization
        """
        super().__init__()
        if input_type == "pointcloud":
            self.proj_input = nn.Linear(902, embedding_dim)
        elif input_type == "pb_id":
            self.proj_input = nn.Linear(num_objects*2, embedding_dim)
        self.rb1 = ResidualBlock(embedding_dim, batchnorm=batchnorm)
        self.rb2 = ResidualBlock(embedding_dim, batchnorm=batchnorm)
        self.rb3 = ResidualBlock(embedding_dim, batchnorm=batchnorm)
        self.proj_output = nn.Linear(embedding_dim, num_objects+1)

        # calculation for running rotated_pc_mean and variance
        # self.input_sum = torch.zeros(902)
        # self.input_std = torch.zeros(902)
        # https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
        self.m_k = None
        self.v_k = None
        self.k = 1
        self.input_type = input_type
        self.normalization_type = normalization_type
        assert normalization_type in [None, "online", "static"]
        # if normalization_type == "static":
        #     assert rotated_pc_mean is not None
        #     assert rotated_pc_var is not None
        #     self.rotated_pc_mean = torch.Tensor(rotated_pc_mean).to(self.device)
        #     self.rotated_pc_var = torch.Tensor(rotated_pc_var).to(self.device)
        self.batchnorm = batchnorm

    def forward(self, batch):
        if self.input_type == "pointcloud":
            nB, nO, num_points, feature_dim = batch['pointcloud'].shape
            pc = batch['pointcloud'].reshape(nB, -1)
            visited = batch['visited'].reshape(nB, -1)
            input_feature_vector = torch.cat((pc, visited), dim=-1)
        elif self.input_type == "pb_id":
            input_feature_vector = torch.cat((batch['occluded_obj_id'], batch['visited']), dim=-1)

        if self.normalization_type == "online":
            if self.m_k is None:
                # [nB, feature_dim] -> [feature_dim]
                self.m_k = input_feature_vector.mean(dim=0)
                self.v_k = 0
            else:
                for sample in input_feature_vector:
                    self.k += 1
                    mk_old = copy.deepcopy(self.m_k)
                    self.m_k = mk_old + (sample - mk_old)/self.k

                    self.v_k = self.v_k + (sample - mk_old)*(sample - self.m_k)

            # normalize input feature vector
            if self.k > 100:
                input_feature_vector = (input_feature_vector - self.m_k)/torch.sqrt((self.v_k / (self.k-1)))
        elif self.normalization_type == "static":
            input_feature_vector = (input_feature_vector - batch['dataset_mean'])/(torch.sqrt(batch['dataset_var']) + 1E-6)
            assert torch.sum(torch.isnan(input_feature_vector)) == 0

        x = self.proj_input(input_feature_vector)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        return self.proj_output(x)




class ProjectAvgPool(nn.Module):
    def __init__(self, embedding_dim, visit_match_xyz=True):
        super().__init__()
        self.embedding_dim = embedding_dim

        if visit_match_xyz:
            self.mlp_fc_initial = nn.Linear(6, embedding_dim)
        else:
            self.mlp_fc_initial = nn.Linear(4, embedding_dim)
        self.mlp_trunk = nn.Sequential(ResidualBlock(embedding_dim), ResidualBlock(embedding_dim), ResidualBlock(embedding_dim))

    def forward(self, pc):
        # project
        x = self.mlp_fc_initial(pc)

        x = self.mlp_trunk(x)
        # average pool
        # nB, num_points, embedding_dim -> nB, embedding_dim
        return x.mean(dim=1)