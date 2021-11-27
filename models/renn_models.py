# process vector input
# concatenate the valid set
# output the selected ID
from .renn_modules import *
from supervised_training.models.pointnet2_models import *
from supervised_training.models.pointnet_models import *
from supervised_training.models.mlp_models import ProjectAvgPool

def process_batch_mlp(batch, append_noop):
    # produce pointcloud vector given batch
    nB, nO, num_points, _ = batch['pointcloud'].shape
    pc = batch['pointcloud'].reshape(nB, nO, num_points*3)

    print("ln28 pc shape")
    print(pc.shape)

    # add noop block
    # nB, nO + 1, num_points*3

    pc = torch.cat((pc, torch.ones(nB, 1, num_points*3).to(pc.device)), dim=1)

    # make tensor of all -1 labels for noops
    noop_visited_labels = torch.ones(nB, 1).to(pc.device) * -1

    # combine visited states with noop visited states
    # [nB, nO] -> [nB, nO + 1, 1]
    visited_reshaped = torch.cat((batch['visited'], noop_visited_labels), dim=-1).unsqueeze(-1)

    # copy the visitation feature num_points*3 times so we have the same dimensionality as PC
    visited_reshaped = visited_reshaped.expand(-1, -1, num_points*3)

    # -> [nB, nO + 1, num_points*3 + 1]
    pc = torch.cat((pc, visited_reshaped), dim=-1)
    return pc


def process_batch_pointnet(batch, append_noop):
    DEVICE = batch['pointcloud'].device
    nB, nO, num_points, _ = batch['pointcloud'].shape

    if append_noop:
        # nB, nO, num_points, 3 -> nB, nO + 1, num_points, 3
        pc = torch.cat((batch['pointcloud'], torch.ones(nB, 1, num_points, 3).to(DEVICE)), dim=1)


        # append visitations
        # nB, nO + 1
        visited = torch.cat((batch['visited'], -1*torch.ones(nB, 1).to(DEVICE)), dim=-1)
    else:
        # nB, nO, num_points, 3
        pc = batch['pointcloud']

        # nB, nO
        visited = batch['visited']

    # nB, nO OR nO + 1, num_points, 1
    visited = visited.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_points, 1)

    # nB, nO OR nO + 1, num_points, 4
    pc = torch.cat((pc, visited), dim=-1)

    # collapse batch and object dimension
    # then permute it
    # -> nB * nO, num_points, 4 -> nB * nO, 4, num_points
    pc = pc.reshape(-1, num_points, 4).permute(0, 2, 1)
    return pc


def process_batch_renn(batch, visit_match_xyz=False):
    """

    :param batch:
    :param visit_match_xyz: Whether to expand the noop parts to match the size of XYZ
    :return:
    """

    DEVICE = batch['pointcloud'].device
    nB, nO, num_points, _ = batch['pointcloud'].shape

    # nB, nO, num_points, 3 -> nB, nO + 1, num_points, 3
    pc = torch.cat((batch['pointcloud'], torch.ones(nB, 1, num_points, 3).to(DEVICE)), dim=1)


    # append visitations
    # nB, nO + 1
    visited = torch.cat((batch['visited'], -1*torch.ones(nB, 1).to(DEVICE)), dim=-1)

    # nB, nO + 1, num_points, 1
    if visit_match_xyz:
        visited = visited.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_points, 3)
    else:
        visited = visited.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_points, 1)

    # nB, nO + 1, num_points, 4 or 6
    pc = torch.cat((pc, visited), dim=-1)

    # -> nB * (nO+1), num_points, 4 or 6
    pc = pc.reshape(-1, num_points, pc.shape[-1])
    return pc


class ReNNIDPredictorSimple(nn.Module):
    def __init__(self, vertex_processing_kwargs=None, embedding_dim=None,
                 normalize=False,
                 p2o_type='linear',
                 p2o_kwargs=None,
                 out_dim=1,
                 append_noop=True):
        super().__init__()
        # skip pointcloud processing...
        # use gnn to process embeddings of the pointclouds
        num_points = 150
        num_feature_dim = 3 # spatial XYZ
        # +1 for the visited label feature dim
        # self.points2object_embedding = nn.Linear(num_points * num_feature_dim + 1, embedding_dim)

        if p2o_type == "linear":
            # the *2 is to account for the visitation
            self.points2object_embedding = nn.Linear(num_points * num_feature_dim * 2, embedding_dim)
        elif p2o_type == "project_and_avgpool":
            self.points2object_embedding = ProjectAvgPool(512)
        elif p2o_type == "pointnet2":
            # set to pointnet2 pytorch
            self.points2object_embedding = Pointnet2ClassificationMSGSmall(embedding_dim-1, normal_channel=False)
        elif p2o_type == "pointnet":
            # set to pointnet2 pytorch
            self.points2object_embedding = PointnetCls(output_dim=embedding_dim, normal_channel=False, **p2o_kwargs)
        elif p2o_type == "renn":
            self.points2object_embedding = VertexProcessingModule(**p2o_kwargs)
        else:
            raise NotImplementedError
        self.p2o_type = p2o_type

        self.object_processor = VertexProcessingModule(**vertex_processing_kwargs)
        # self.object_embedding_to_scalar = nn.Sequential(nn.Linear(embedding_dim, 1))
        self.object_embedding_to_out = nn.Sequential(ResidualBlock(embedding_dim),
                                                     ResidualBlock(embedding_dim),
                                                     nn.Linear(embedding_dim, out_dim))

        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.append_noop = append_noop

    def forward(self, batch):
        # nB, nO, num_points, 3 -> nB, nO, num_points * 3
        if self.p2o_type == "linear":
            pc = process_batch_mlp(batch)

            if self.normalize:
                pc = (pc - batch['dataset_mean'])/(torch.sqrt(batch['dataset_var']) + 1E-6)
                assert torch.sum(torch.isnan(pc)) == 0

            oe = self.points2object_embedding(pc)
        elif self.p2o_type == "pointnet2":
            nB, nO, num_points, _ = batch['pointcloud'].shape

            # nB, nO, num_points, 3 -> oe: nB * nO, embedding_dim
            # oe, l3_points = self.points2object_embedding(batch['pointcloud'].reshape(-1, num_points, 3).permute(0, 2, 1))
            oe = self.points2object_embedding(batch['pointcloud'].reshape(-1, num_points, 3).permute(0, 2, 1))

            oe = oe.reshape(nB, nO, -1)

            # concatenate noop block
            # -> nB, nO + 1, oe_embedding_dim
            oe = torch.cat((oe, torch.ones(nB, 1, oe.shape[-1]).to(oe.device)), dim=1)

            # append visitations
            # add noop visitation of -1
            # nB, nO -> nB, nO + 1 -> nB, nO + 1, 1
            visited = torch.cat((batch['visited'], torch.ones(nB, 1).to(oe.device) * -1), dim=-1).unsqueeze(-1)

            oe = torch.cat((oe, visited), dim=-1)
        elif self.p2o_type == "pointnet":
            nB, nO, num_points, _ = batch['pointcloud'].shape

            pc = process_batch_pointnet(batch, self.append_noop)

            if self.normalize:
                pc = (pc - batch['dataset_mean'].unsqueeze(0).unsqueeze(-1))/(torch.sqrt(batch['dataset_var'].unsqueeze(0).unsqueeze(-1)) + 1E-6)

            # -> nB * (nO + 1), embedding_dim
            oe = self.points2object_embedding(pc)

            if self.append_noop:
                # -> nB, nO + 1, embedding_dim
                oe = oe.reshape(nB, nO + 1, self.embedding_dim)
            else:
                oe = oe.reshape(nB, nO, self.embedding_dim)

        elif self.p2o_type == "renn":
            nB, nO, num_points, _ = batch['pointcloud'].shape

            pc = process_batch_renn(batch)

            if self.normalize:
                pc = (pc - batch['dataset_mean'].unsqueeze(0).unsqueeze(0))/\
                     (torch.sqrt(batch['dataset_var'].unsqueeze(0).unsqueeze(0)) + 1E-6)

            # -> nB * (nO + 1), embedding_dim
            oe = self.points2object_embedding(pc)

            # -> nB, nO + 1, embedding_dim
            oe = oe.reshape(nB, nO + 1, self.embedding_dim)
        elif self.p2o_type == "project_and_avgpool":
            nB, nO, num_points, _ = batch['pointcloud'].shape
            pc = process_batch_renn(batch, visit_match_xyz=True)

            if self.normalize:
                pc = (pc - batch['dataset_mean'].unsqueeze(0).unsqueeze(0))/ \
                     (torch.sqrt(batch['dataset_var'].unsqueeze(0).unsqueeze(0)) + 1E-6)

            oe = self.points2object_embedding(pc)
            oe = oe.reshape(nB, nO + 1, self.embedding_dim)
        else:
            raise NotImplementedError
        oe_clone = oe.clone()

        # -> nB, nO, embedding_dim
        object_embeddings = self.object_processor(self.ln1(oe))

        if self.object_processor.no_pool:
            object_embeddings = oe_clone + object_embeddings

            # add a noop block
            # noop_block = torch.ones(nB, 1, self.embedding_dim).to(object_embeddings.device)

            # normalize noop block
            # if self.normalize:
            #     noop_block = batch['dataset_mean']

            # object_embeddings_w_noop = torch.cat((object_embeddings, noop_block), dim=1)
            # return self.object_embedding_to_scalar(object_embeddings_w_noop)
        return self.object_embedding_to_out(object_embeddings)


