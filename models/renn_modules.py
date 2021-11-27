from bandu.imports.rlkit.torch.core import PyTorchModule
from torch import nn, nn as nn
import torch
import torch.nn.functional as F
import pprint
from torch.nn import Parameter
from bandu.imports.rlkit.torch.networks import Mlp
import bandu.imports.rlkit.torch.pytorch_util as ptu


class Attention(PyTorchModule):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 query_key_activation_fnx=None,
                 softmax_temperature=1.0,
                 interaction_type="sum"):
        assert query_key_activation_fnx is not None
        self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.query_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))
        self.query_key_activation_fnx = query_key_activation_fnx
        self.activation_fnx = activation_fnx
        self.interaction_type = interaction_type
        assert interaction_type in ['sum', 'mlp']
        if interaction_type == "mlp":
            self.fc_interaction = nn.Linear(2*embedding_dim, embedding_dim)

    def forward(self, query, context, memory, mask, reduce_heads=True, return_probs=False):
        """
        N, nV, nE memory -> N, nV, nE updated memory

        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = query.size()
        # assert len(query.size()) == 3

        # assert self.fc_createheads.out_features % nE == 0
        nH = int(self.fc_createheads.out_features / nE)

        nV = memory.size(1)

        # assert len(mask.size()) == 2

        # N, nQ, nE -> N, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nQ, nH, nE)
        # else:
        #     query = query.view(N, nQ, nH, nE)

        if self.query_norm is not None:
            query = self.query_norm(query)
        # if self.layer_norms is not None:
        # query = self.layer_norms[0](query)
        # N, nQ, nH, nE -> N, nQ, nV, nH, nE
        query = query.unsqueeze(2).expand(-1, -1, nV, -1, -1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)

        # -> N, nQ, nV, nH, 1
        # qc_logits = self.fc_logit(torch.tanh(context + query))

        if self.interaction_type == 'mlp':
            hidden = self.fc_interaction(torch.cat((context, query), dim=-1))
            qc_logits = self.fc_logit(self.query_key_activation_fnx(hidden))
        else:
            qc_logits = self.fc_logit(self.query_key_activation_fnx(context + query))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)

        if return_probs:
            ret_attention_probs = attention_probs.squeeze(-1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(2).squeeze(2)

        if not reduce_heads:
            return [attention_heads]

        attention_heads = self.activation_fnx(attention_heads)
        # N, nQ, nH, nE -> N, nQ, nE
        # if nQ > 1:
        attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nH*nE))
        # else:
        #     attention_result = attention_heads.view(N, nQ, nE)

        # attention_result = self.activation_fnx(attention_result)
        #TODO: add nonlinearity here...

        # if self.layer_norms is not None:
        #     attention_result = self.layer_norms[2](attention_result)

        # assert len(attention_result.size()) == 3

        if return_probs:
            return [attention_result, ptu.get_numpy(ret_attention_probs)]
        else:
            return [attention_result]

class VertexProcessingModule(nn.Module):
    def __init__(self,
                 embedding_dim=64,
                 num_relational_blocks=3,
                 num_mlp_layers=3,
                 layer_norm=False,
                 mlp_layer_norm=False,
                 num_heads=1,
                 pooler_layer_norm=False,
                 attention_kwargs=None,
                 num_qcm_modules=1,
                 input_dim=3,
                 no_pool=False,
                 multiobject_attention=False,
                 concat_input_vertices=False,
                 *args,
                 **kwargs):
        """
        [nB, nO, num_points, embedding_dim] -> [nB, nO, num_points, embedding_dim]
        :param embedding_dim:
        :param num_relational_blocks:
        :param num_mlp_layers:
        :param layer_norm:
        :param mlp_layer_norm:
        :param num_heads:
        :param pooler_layer_norm:
        :param attention_kwargs:
        :param num_qcm_modules:
        :param input_dim:
        :param no_pool:
        :param multiobject_attention:
        :param args:
        :param kwargs:
        """
        super().__init__()
        assert attention_kwargs is not None, pprint.pprint(kwargs)
        self.first_fc_embed = nn.Linear(input_dim, embedding_dim)

        if multiobject_attention:
            self.g2g_list = nn.ModuleList([AttentiveGraphToGraph(num_qcm_modules=num_qcm_modules,
                                                                 embedding_dim=embedding_dim,
                                                                 num_heads=num_heads,
                                                                 multiobject_attention=multiobject_attention,
                                                                 layer_norm=layer_norm,
                                                                 attention_kwargs=attention_kwargs) for i in range(num_relational_blocks)])
        else:
            self.g2g_list = nn.ModuleList([AttentiveGraphToGraph(num_qcm_modules=num_qcm_modules,
                                                                 embedding_dim=embedding_dim,
                                                                 num_heads=num_heads,
                                                                 layer_norm=layer_norm,
                                                                 attention_kwargs=attention_kwargs) for i in range(num_relational_blocks)])
        self.g2g_output_norm_list = nn.ModuleList([nn.LayerNorm(embedding_dim) if layer_norm else None for i in range(num_relational_blocks)])

        if not no_pool:
            self.pooler = AttentiveGraphPooling(attention_kwargs=attention_kwargs,
                                                embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                layer_norm=layer_norm)
            self.pooler_output_norm = nn.LayerNorm(2 * embedding_dim) if pooler_layer_norm else None
            self.mlp_proj = nn.Linear(2 * embedding_dim, embedding_dim)
            self.mlp_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_mlp_layers)])
            self.mlp_norm_list= nn.ModuleList([nn.LayerNorm(embedding_dim) if mlp_layer_norm else None for i in range(num_mlp_layers)])
        self.no_pool = no_pool
        self.multiobject_attention = multiobject_attention
        self.concat_input_vertices = concat_input_vertices

    def forward(self, vertices):
        # If this line errors, check that all the concatenate_xxx_idx are set correctly
        # Or, check that the classifier is concatenating the bin indices correctly
        assert list(self.first_fc_embed.parameters())[0].is_cuda
        assert vertices.is_cuda
        hidden_vertices = self.first_fc_embed(vertices)

        if self.concat_input_vertices:
            # represents each object
            input_vertices = hidden_vertices.clone()

        for i in range(len(self.g2g_list)):
            mask = torch.ones_like(hidden_vertices)[:, :, 0]
            # hidden_vertices = F.leaky_relu(self.g2g_list[i](hidden_vertices,
            #                                                 mask)[0] + hidden_vertices)
            if self.concat_input_vertices:
                hidden_vertices = self.g2g_list[i](input_vertices + hidden_vertices,
                                                                mask)[0] + hidden_vertices
            else:
                hidden_vertices = self.g2g_list[i](hidden_vertices,
                                                   mask)[0] + hidden_vertices
            # if self.g2g_output_norm_list[i] is not None:
            #     hidden_vertices = self.g2g_output_norm_list[i](hidden_vertices)

        if self.no_pool:
            return hidden_vertices

        mask = torch.ones_like(hidden_vertices)[..., 0]
        attention_pooled_embedding = self.pooler(hidden_vertices, mask)[0]

        num_points_dimension = 2 if self.multiobject_attention else 1
        max_pooled_embedding = torch.max(hidden_vertices, dim=num_points_dimension)[0]

        pooled_embedding = torch.cat((attention_pooled_embedding, max_pooled_embedding), dim=-1)

        if self.pooler_output_norm is not None:
            pooled_embedding = self.pooler_output_norm(pooled_embedding)
        # return pooled_embedding
        embedding = self.mlp_proj(pooled_embedding)

        for i in range(len(self.mlp_list)):
            embedding = self.mlp_list[i](embedding) + embedding
            if self.mlp_norm_list[i] is not None:
                embedding = self.mlp_norm_list[i](embedding)
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim, layernorm=True, activation_fn="leaky_relu"):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)

        if activation_fn == "leaky_relu":
            self.activation_fn = F.leaky_relu
        elif activation_fn == "tanh":
            self.activation_fn = torch.tanh
        else:
            raise NotImplementedError
        if layernorm:
            self.layernorm = layernorm
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, vertices):
        if self.layernorm:
            return vertices + self.activation_fn(self.linear(self.ln(vertices)))
        else:
            return vertices + self.activation_fn(self.linear(vertices))


class AttentiveGraphToGraph(PyTorchModule):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True,
                 attention_kwargs=None,
                 num_qcm_modules=1,
                 multiobject_attention=False,
                 **kwargs):
        def lin_module_gen(embedding_dim):
            return nn.Sequential(nn.Linear(embedding_dim,embedding_dim),
                                 nn.LeakyReLU(),
                                 nn.LayerNorm(embedding_dim))
        if attention_kwargs is None:
            attention_kwargs = dict()
        self.save_init_params(locals())
        super().__init__()

        self.vertex_mlp_fc_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_qcm_modules)])
        self.vertex_mlp_norm_list = nn.ModuleList([nn.LayerNorm(embedding_dim) if layer_norm else None for i in range(num_qcm_modules)])
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)

        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)
        self.multiobject_attention = multiobject_attention

    def forward(self, vertices, mask, **kwargs):
        """

        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        input_vertices = vertices.clone()
        for i in range(len(self.vertex_mlp_fc_list)):
            vertices = F.leaky_relu(self.vertex_mlp_fc_list[i](vertices) + vertices)
            if self.vertex_mlp_norm_list[i] is not None:
                vertices = self.vertex_mlp_norm_list[i](vertices)

        # residual
        vertices = input_vertices + vertices

        input_vertices2 = vertices.clone()

        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return [self.attention(query, context, memory, mask, **kwargs)[0] + input_vertices2]


class AttentiveGraphPooling(PyTorchModule):
    """
    Pools nV vertices to a single vertex embedding

    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True,
                 mlp_kwargs=None,
                 attention_kwargs=None,
                 multiobject_attention=None):
        # assert num_objects is not None, "You must pass in num_objects"
        self.save_init_params(locals())
        super().__init__()
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None
        # self.num_objects = num_objects
        self.multiobject_attention = multiobject_attention

    def forward(self, vertices, mask, **kwargs):
        """
        N, nO, nV, nE -> N, nO, nE
        :param vertices:
        :param mask:
        :return: list[attention_result]
        """
        if self.multiobject_attention:
            N, nO, nV, nE = vertices.size()
            # nE -> N, nO, nQ, nE where nQ == self.num_heads
            nQ = 1
            query = self.input_independent_query.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, nO, nQ, -1)
        else:
            N, nV, nE = vertices.size()
            # nE -> N, nQ, nE where nQ == self.num_heads
            query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)


        if self.layer_norm is not None:
            cm_block = self.layer_norm(self.fc_cm(vertices))
        else:
            cm_block = self.fc_cm(vertices)
        context, memory = cm_block.chunk(2, dim=-1)

        attention_out = self.attention(query, context, memory, mask, **kwargs)

        attention_result = attention_out[0]
        if 'reduce_heads' in kwargs and not kwargs['reduce_heads']:
            assert len(attention_result.shape) == 4
            return [attention_result]

        squeeze_idx = 1
        if self.proj is not None:
            ret_out = [self.proj(attention_result).squeeze(squeeze_idx)]
        else:
            ret_out = [attention_result.squeeze(squeeze_idx)]

        if 'return_probs' in kwargs and kwargs['return_probs']:
            ret_out.append(attention_out[1])
        return ret_out


# class AttentiveGraphToGraphPrenorm(PyTorchModule):
#     """
#     Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
#     """
#     def __init__(self,
#                  embedding_dim=64,
#                  num_heads=1,
#                  attention_kwargs=None,
#                  num_qcm_modules=1,
#                  multiobject_attention=False,
#                  **kwargs):
#
#         if attention_kwargs is None:
#             attention_kwargs = dict()
#         self.save_init_params(locals())
#         super().__init__()
#
#         self.ffn = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_qcm_modules)])
#
#         self.prenorm_attention = nn.LayerNorm(embedding_dim)
#
#         self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
#         # self.fc_qcm = nn.Sequential(*[lin_module_gen(embedding_dim) for i in range(num_qcm_modules)],
#         #                             nn.Linear(embedding_dim, 3 * embedding_dim))
#         self.attention = Attention(embedding_dim, num_heads=num_heads, **attention_kwargs)
#
#         # self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None
#
#         self.prenorm_ffn = nn.LayerNorm(embedding_dim)
#
#
#     def forward(self, vertices, mask, **kwargs):
#         """
#
#         :param vertices: N x nV x nE
#         :return: updated vertices: N x nV x nE
#         """
#         # norm vertices before it goes into the attention block
#         vertices_normed = self.prenorm_attention(vertices)
#
#         qcm_block = self.fc_qcm(vertices_normed)
#
#         query, context, memory = qcm_block.chunk(3, dim=-1)
#
#         # attention + residual connection
#         # [0] is because attention returns a list (where [1] might be diagnostics)
#         vertices = self.attention(query, context, memory, mask, **kwargs)[0] + vertices
#
#         # norm before we go into the MLP
#         vertices_normed = self.prenorm_ffn(vertices)
#         for i in range(len(self.ffn)):
#             vertices_normed = F.leaky_relu(self.ffn[i](vertices_normed) + vertices_normed)
#
#         # residual between MLP and pre-MLP activations
#         vertices = vertices_normed + vertices
#         return [vertices]