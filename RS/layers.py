from itertools import combinations, product

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class Dice(nn.Module):
    """
    activation function DICE in DIN
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9

    def forward(self, x):
        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1 - p) + x.mul(p)
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, fc_dims, input_dim, dropout):
        super(MLP, self).__init__()
        fc_layers = []
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc(x)


class MoE(nn.Module):
    """
    Mixture of Export
    """
    def __init__(self, moe_arch, inp_dim, dropout):
        super(MoE, self).__init__()
        export_num, export_arch = moe_arch
        self.export_num = export_num
        self.gate_net = nn.Linear(inp_dim, export_num)
        self.export_net = nn.ModuleList([MLP(export_arch, inp_dim, dropout) for _ in range(export_num)])

    def forward(self, x):
        gate = self.gate_net(x).view(-1, self.export_num)  # (bs, export_num)
        gate = nn.functional.softmax(gate, dim=-1).unsqueeze(dim=1) # (bs, 1, export_num)
        experts = [net(x) for net in self.export_net]
        experts = torch.stack(experts, dim=1)  # (bs, expert_num, emb)
        out = torch.matmul(gate, experts).squeeze(dim=1)
        return out


class HEA(nn.Module):
    """
    hybrid-expert adaptor
    """
    def __init__(self, ple_arch, inp_dim, dropout):
        super(HEA, self).__init__()
        share_expt_num, spcf_expt_num, expt_arch, task_num = ple_arch
        self.share_expt_net = nn.ModuleList([MLP(expt_arch, inp_dim, dropout) for _ in range(share_expt_num)])
        self.spcf_expt_net = nn.ModuleList([nn.ModuleList([MLP(expt_arch, inp_dim, dropout)
                                                           for _ in range(spcf_expt_num)]) for _ in range(task_num)])
        self.gate_net = nn.ModuleList([nn.Linear(inp_dim, share_expt_num + spcf_expt_num)
                                   for _ in range(task_num)])

    def forward(self, x_list):
        gates = [net(x) for net, x in zip(self.gate_net, x_list)]
        gates = torch.stack(gates, dim=1)  # (bs, tower_num, expert_num), export_num = share_expt_num + spcf_expt_num
        gates = nn.functional.softmax(gates, dim=-1).unsqueeze(dim=2)  # (bs, tower_num, 1, expert_num)
        cat_x = torch.stack(x_list, dim=1)  # (bs, tower_num, inp_dim)
        share_experts = [net(cat_x) for net in self.share_expt_net]
        share_experts = torch.stack(share_experts, dim=2)  # (bs, tower_num, share_expt_num, E)
        spcf_experts = [torch.stack([net(x) for net in nets], dim=1)
                        for nets, x in zip(self.spcf_expt_net, x_list)]
        spcf_experts = torch.stack(spcf_experts, dim=1)  # (bs, tower_num, spcf_expt_num, num)
        experts = torch.cat([share_experts, spcf_experts], dim=2)  # (bs, tower_num, expert_num, E)
        export_mix = torch.matmul(gates, experts).squeeze(dim=2)  # (bs, tower_num, E)
        # print('export mix', export_mix.shape, 'tower num', self.tower_num)
        export_mix = torch.split(export_mix, dim=1, split_size_or_sections=1)
        out = [x.squeeze(dim=1) for x in export_mix]
        return out


class ConvertNet(nn.Module):
    """
    convert from semantic space to recommendation space
    """
    def __init__(self, args, inp_dim, dropout, conv_type):
        super(ConvertNet, self).__init__()
        self.type = conv_type
        self.device = args.device
        print(self.type)
        if self.type == 'MoE':
            print('convert module: MoE')
            moe_arch = args.export_num, args.convert_arch
            self.sub_module = MoE(moe_arch, inp_dim, dropout)
        elif self.type == 'HEA':
            print('convert module: HEA')
            ple_arch = args.export_num, args.specific_export_num, args.convert_arch, args.augment_num
            self.sub_module = HEA(ple_arch, inp_dim, dropout)
        else:
            print('convert module: MLP')
            self.sub_module = MLP(args.convert_arch, inp_dim, dropout).to(self.device)

    def forward(self, x_list):
        if self.type == 'HEA':
            out = self.sub_module(x_list)
        else:
            out = [self.sub_module(x) for x in x_list]
        out = torch.cat(out, dim=-1)
        return out


class AttentionPoolingLayer(nn.Module):
    """
      attention pooling in DIN
    """

    def __init__(self, embedding_dim, dropout, fc_dims=[32, 16]):
        super(AttentionPoolingLayer, self).__init__()
        fc_layers = []
        input_dim = embedding_dim * 4
        # fc layer
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p=dropout))
            input_dim = fc_dim

        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query, user_behavior, mask=None):
        """
          :param query_ad:embedding of target item   -> (bs, dim)
          :param user_behavior:embedding of user behaviors     ->  (bs, seq_len, dim)
          :param mask:mask on user behaviors  ->  (bs,seq_len, 1)
          :return output:user interest (bs, dim)
        """
        query = query.unsqueeze(1)
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior,
                                queries * user_behavior], dim=-1)
        attns = self.fc(attn_input)
        if mask is not None:
            attns = attns.mul(mask)
        out = user_behavior.mul(attns)
        output = out.sum(dim=1)
        return output, attns


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** :dimension of input feature
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: cross net
        - **mode**: "v1"  or "v2" ,DCNv1 or DCNv2
    """

    def __init__(self, inp_dim, layer_num=2, mode='v1'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.mode = mode
        if self.mode == 'v1': # DCN
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList([
                nn.Parameter(nn.init.xavier_normal_(torch.empty(inp_dim, 1))) for _ in range(self.layer_num)])
        elif self.mode == 'v2': # DCNv2
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([
                nn.Parameter(nn.init.xavier_normal_(torch.empty(inp_dim, inp_dim))) for _ in range(self.layer_num)])
        else:  # error
            raise ValueError("mode in CrossNet should be 'v1' or 'v2'")

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(inp_dim, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.mode == 'v1':
            # x0 * (xl.T * w )+ b + xl
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.mode == 'v2':
            # x0 * (Wxl + bl) + xl
                dot_ = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                raise ValueError("mode in CrossNet should be 'v1' or 'v2'")

            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units):
        super(CIN, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, X_0):
        pooling_outputs = []
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        return concate_vec


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0.,
                 use_residual=True, use_scale=False, layer_norm=False, align_to="input"):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            if align_to == "output":
                self.W_res = nn.Linear(input_dim, self.output_dim, bias=False)
            elif align_to == "input":
                self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query

        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            if self.align_to == "output":  # AutoInt style
                residual = self.W_res(residual)
            elif self.align_to == "input":  # Transformer stype
                output = self.W_res(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class SqueezeExtractionLayer(nn.Module):
    def __init__(self, num_fields, reduction_ratio):
        super(SqueezeExtractionLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(nn.Module):
    def __init__(self, embed_size, num_fields, bilinear_type):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(embed_size, embed_size, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for _ in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False)
                                                 for _, _ in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class GraphLayer(nn.Module):
    def __init__(self, num_fields, embed_size):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embed_size, embed_size))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embed_size, embed_size))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embed_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)  # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNNBlock(nn.Module):
    def __init__(self, num_fields, embed_size, gnn_layer_num, res_conn, reuse_graph_layer):
        super(FiGNNBlock, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embed_size
        self.gnn_layers = gnn_layer_num
        self.use_residual = res_conn
        self.reuse_graph_layer = reuse_graph_layer
        if self.reuse_graph_layer:
            self.gnn = GraphLayer(self.num_fields, self.embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(self.num_fields, self.embedding_dim) for _ in range(self.gnn_layers)])
        self.gru = nn.GRUCell(embed_size, embed_size)
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embed_size * 2, 1, bias=False)

    def build_graph_with_attention(self, feat_embed):
        src_emb = feat_embed[:, self.src_nodes, :]
        dst_emb = feat_embed[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        try:
            device = feat_embed.get_device()
            mask = torch.eye(self.num_fields).to(device)
        except RuntimeError:
            mask = torch.eye(self.num_fields)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = nn.functional.softmax(alpha, dim=-1)  # batch x field x field without self-loops
        return graph

    def forward(self, feat_embed):
        g = self.build_graph_with_attention(feat_embed)
        h = feat_embed
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feat_embed
        return h


class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embed_size):
        super(AttentionalPrediction, self).__init__()
        self.linear1 = nn.Linear(embed_size, 1, bias=False)
        self.linear2 = nn.Sequential(nn.Linear(num_fields * embed_size, num_fields, bias=False),
                                     nn.Sigmoid())

    def forward(self, h):
        score = self.linear1(h).squeeze(-1)
        weight = self.linear2(h.flatten(start_dim=1))
        logits = (weight * score).sum(dim=1).unsqueeze(-1)
        return logits


class InterestExtractor(nn.Module):
    """
    Interest extractor in DIEN
    """
    def __init__(self, input_size, hidden_size):
        super(InterestExtractor, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, keys, keys_length):
        """
        keys:        [btz, seq_len, hdsz]
        keys_length: [btz, 1]
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        if keys_length.shape[0] == 0:
            return torch.zeros(btz, hdsz, device=keys.device)

        masked_keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)  # 去除全为0序列的样本
        packed_keys = pack_padded_sequence(masked_keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0, total_length=seq_len)

        return interests


class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)
        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)


class InterestEvolving(nn.Module):
    """
    Interest evolving in DIEN
    """

    def __init__(self, input_size, gru_type='GRU', dropout=0):
        super(InterestEvolving, self).__init__()
        assert gru_type in {'GRU', 'AIGRU', 'AGRU', 'AUGRU'}, f"gru_type: {gru_type} is not supported"
        self.gru_type = gru_type

        if gru_type == 'GRU':
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size, gru_type=gru_type)

        self.attention = AttentionPoolingLayer(embedding_dim=input_size, dropout=dropout)


    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        query:       [btz, 1, hdsz]
        keys:        [btz, seq_len ,hdsz]
        keys_length: [btz, 1]
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        zero_outputs = torch.zeros(btz, hdsz, device=query.device)
        if keys_length.shape[0] == 0:
            return zero_outputs

        query = torch.masked_select(query, smp_mask.view(-1, 1)).view(-1, hdsz)
        keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)  # 去除全为0序列的样本

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0,
                                               total_length=seq_len)
            outputs, _ = self.attention(query, interests, mask)  # [btz1, hdsz]

        elif self.gru_type == 'AIGRU':
            _, att_scores = self.attention(query, keys, mask)  # [btz1, 1, seq_len]
            interests = keys * att_scores # [btz1, seq_len, hdsz]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [btz1, hdsz]

        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            _, att_scores = self.attention(query, keys, mask) # [b, T]
            att_scores = att_scores.squeeze(1)
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True,
                                                    enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True,
                                                 enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=seq_len)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length)  # [b, H]

        # [b, H] -> [B, H]
        zero_outputs[smp_mask.squeeze(1)] = outputs
        return zero_outputs


class Phi_function(nn.Module):
    """
    phi function on
    """

    def __init__(self, input_size, hidden_size, dropout=0):
        super(Phi_function, self).__init__()
        self.bn1 = torch.nn.BatchNorm1d(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.tanh = torch.nn.Tanh()
        self.dp1 = torch.nn.Dropout(dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)

    def forward(self, seq_state, final_state):
        bn1 = self.bn1(final_state)
        fc1 = self.fc1(bn1)
        dp1 = self.dp1(self.tanh(fc1))
        bn2 = self.bn2(seq_state.transpose(1, 2)).transpose(1, 2)
        fc2 = self.fc2(torch.unsqueeze(dp1, dim=1) * bn2)
        score = torch.softmax(fc2, dim=-1)
        seq_len = score.shape[1]
        score = score[:, :, 0].view([-1, seq_len])
        return score


class SLAttention(nn.Module):
    """
    SLAttention for MIR
    v for item, q for hist
    """
    def __init__(self, v_fnum, q_fnum, emb_dim, v_len, q_len, fi=True, ii=True):
        super(SLAttention, self).__init__()
        self.v_fnum = v_fnum
        self.q_fnum = q_fnum
        self.emb_dim = emb_dim
        self.v_dim = v_fnum * emb_dim
        self.q_dim = q_fnum * emb_dim
        self.v_len = v_len
        self.q_len = q_len
        self.fi = fi
        self.ii = ii
        if fi:
            self.w_b_fi = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.emb_dim, self.emb_dim)))
            self.fi_conv = nn.Conv2d(1, 1, kernel_size=(self.q_fnum, self.v_fnum), stride=(self.q_fnum, self.v_fnum))
        if ii:
            self.w_b_ii = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.q_dim, self.v_dim)))

        self.w_v = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.v_dim, self.v_len)))
        self.w_q = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, self.q_dim, self.v_len)))

    def forward(self, V, Q):
        batch_size = V.shape[0]
        if self.ii:
            w_b_ii = self.w_b_ii.repeat([batch_size, 1, 1])
            V_trans = V.permute(0, 2, 1)
            C1 = torch.matmul(Q, torch.matmul(w_b_ii, V_trans))
            C = torch.tan(C1)
        if self.fi:
            V_s = V.view(-1, self.v_len * self.v_fnum, self.emb_dim)
            Q_s = Q.view(-1, self.q_len * self.q_fnum, self.emb_dim)
            w_b_fi = self.w_b_fi.repeat([batch_size, 1, 1])
            V_s_trans = V_s.permute(0, 2, 1)
            C2 = torch.matmul(Q_s, torch.matmul(w_b_fi, V_s_trans)).unsqueeze(1)

            C2 = self.fi_conv(C2)
            C2 = C2.view(-1, self.q_len, self.v_len)
            if self.ii:
                C = torch.tanh(C1 + C2)
            else:
                C = torch.tanh(C2)

        hv_1 = torch.matmul(V.view(-1, self.v_dim), self.w_v).view(-1, self.v_len, self.v_len)
        hq_1 = torch.matmul(Q.view(-1, self.q_dim), self.w_q).view(-1, self.q_len, self.v_len)
        hq_1 = hq_1.permute(0, 2, 1)
        h_v = torch.tanh(hv_1 + torch.matmul(hq_1, C))
        h_q = torch.tanh(hq_1 + torch.matmul(hv_1, C.permute(0, 2, 1)))
        a_v = torch.softmax(h_v, dim=-1)
        a_q = torch.softmax(h_q, dim=-1)
        v = torch.matmul(a_v, V)
        q = torch.matmul(a_q, Q)
        return v, q, a_v, a_q
