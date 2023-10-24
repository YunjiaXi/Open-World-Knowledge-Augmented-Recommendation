'''
-*- coding: utf-8 -*-
@File  : models.py
'''
import numpy as np
import torch
import torch.nn as nn
from layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention
from layers import Phi_function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class BaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()
        self.task = args.task
        self.args = args
        self.augment_num = 2 if args.augment else 0
        args.augment_num = self.augment_num

        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        self.rating_num = dataset.rating_num
        self.dense_dim = dataset.dense_dim
        self.max_hist_len = args.max_hist_len
        if self.task == 'rerank':
            self.max_list_len = dataset.max_list_len

        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.rnn_dp = args.rnn_dp
        self.output_dim = args.output_dim
        self.convert_dropout = args.convert_dropout
        self.convert_type = args.convert_type
        self.auxiliary_loss_weight = args.auxi_loss_weight

        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim
        self.dens_vec_num = 0

        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim)
        self.attr_embedding = nn.Embedding(self.attr_num + 1, self.embed_dim)
        self.rating_embedding = nn.Embedding(self.rating_num + 1, self.embed_dim)
        if self.augment_num:
            self.convert_module = ConvertNet(args, self.dense_dim, self.convert_dropout, self.convert_type)
            self.dens_vec_num = args.convert_arch[-1] * self.augment_num

        self.module_inp_dim = self.get_input_dim()
        self.field_num = self.get_field_num()
        self.convert_loss = 0

    def process_input(self, inp):
        device = next(self.parameters()).device
        hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
        hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                 self.embed_dim * self.attr_fnum)
        hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                      self.embed_dim)
        hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
        hist_len = inp['hist_seq_len'].to(device)

        if self.task == 'ctr':
            iid_emb = self.item_embedding(inp['iid'].to(device))
            attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            # item_emb = item_emb.view(-1, self.itm_emb_dim)
            labels = inp['lb'].to(device)
            if self.augment_num:
                orig_dens_vec = [inp['hist_aug_vec'].to(device), inp['item_aug_vec'].to(device)]
                dens_vec = self.convert_module(orig_dens_vec)
            else:
                dens_vec, orig_dens_vec = None, None
            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_vec, labels
        elif self.task == 'rerank':
            iid_emb = self.item_embedding(inp['iid_list'].to(device))
            attr_emb = self.attr_embedding(inp['aid_list'].to(device)).view(-1, self.max_list_len,
                                                                            self.embed_dim * self.attr_fnum)
            item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
            item_emb = item_emb.view(-1, self.max_list_len, self.itm_emb_dim)
            labels = inp['lb_list'].to(device).view(-1, self.max_list_len)
            if self.augment_num:
                hist_aug = inp['hist_aug_vec'].to(device)
                item_list_aug = inp['item_aug_vec_list']
                orig_dens_list = [[hist_aug, item_aug.to(device)] for item_aug in item_list_aug]
                dens_vec_list = [self.convert_module(orig_dens) for orig_dens in orig_dens_list]
                dens_vec = torch.stack([dens for dens in dens_vec_list], dim=1)
            else:
                dens_vec, orig_dens_list = None, None

            return item_emb, hist_emb, hist_len, dens_vec, orig_dens_list, labels
        else:
            raise NotImplementedError

    def get_input_dim(self):
        if self.task == 'ctr':
            return self.hist_emb_dim + self.itm_emb_dim + self.dens_vec_num
        elif self.task == 'rerank':
            return self.itm_emb_dim + self.dens_vec_num
        else:
            raise NotImplementedError

    def get_field_num(self):
        return self.item_fnum + self.augment_num + self.hist_fnum

    def get_filed_input(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels

    def process_rerank_inp(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)

        if self.augment_num:
            out = torch.cat([item_embedding, dens_vec], dim=-1)
        else:
            out = item_embedding
        return out, labels

    def get_ctr_output(self, logits, labels=None):
        outputs = {
            'logits': torch.sigmoid(logits),
            'labels': labels,
        }

        if labels is not None:
            if self.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.output_dim)), labels.float())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight

        return outputs

    def get_rerank_output(self, logits, labels=None, attn=False):
        outputs = {
            'logits': logits,
            'labels': labels,
        }

        if labels is not None:
            if attn:
                logits = attention_score(logits.view(-1, self.max_list_len), self.args.temperature)
                labels = attention_score(labels.float().view(-1, self.max_list_len), self.args.temperature)
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs['loss'] = loss + self.convert_loss * self.auxiliary_loss_weight
        return outputs

    def get_mask(self, length, max_len):
        device = next(self.parameters()).device
        rang = torch.arange(0, max_len).view(-1, max_len).to(device)
        batch_rang = rang.repeat([length.shape[0], 1])
        mask = batch_rang < torch.unsqueeze(length, dim=-1)
        return mask.unsqueeze(dim=-1).long()


class DeepInterestNet(BaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DeepInterestNet, self).__init__(args, dataset)

        self.map_layer = nn.Linear(self.hist_emb_dim, self.itm_emb_dim)
        # embedding of history item and candidate item should be the same
        self.attention_net = AttentionPoolingLayer(self.itm_emb_dim, self.dropout)

        # history embedding, item embedding, and user embedding
        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        mask = self.get_mask(hist_len, self.max_hist_len)

        user_behavior = self.map_layer(user_behavior)
        user_interest, _ = self.attention_net(query, user_behavior, mask)

        if self.augment_num:
            concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
        else:
            concat_input = torch.cat([user_interest, query], dim=-1)

        mlp_out = self.final_mlp(concat_input)
        logits = self.final_fc(mlp_out)
        out = self.get_ctr_output(logits, labels)
        return out


class DIEN(BaseModel):
    """
    DIN
    """

    def __init__(self, args, dataset):
        super(DIEN, self).__init__(args, dataset)

        self.interest_extractor = InterestExtractor(self.hist_emb_dim, self.itm_emb_dim)
        self.interest_evolution = InterestEvolving(self.itm_emb_dim, gru_type=args.dien_gru, dropout=self.dropout)

        self.final_mlp = MLP(self.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)

    def get_input_dim(self):
        return self.itm_emb_dim * 2 + self.dens_vec_num

    def forward(self, inp):
        """
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        """
        query, user_behavior, length, dens_vec, orig_dens_vec, labels = self.process_input(inp)
        mask = self.get_mask(length, self.max_hist_len)
        length = torch.unsqueeze(length, dim=-1)
        masked_interest = self.interest_extractor(user_behavior, length)
        user_interest = self.interest_evolution(query, masked_interest, length, mask)  # [btz, hdsz]
        # user_interest = masked_interest.sum(dim=1)
        if self.augment_num:
            concat_input = torch.cat([user_interest, query, dens_vec], dim=-1)
        else:
            concat_input = torch.cat([user_interest, query], dim=-1)
        mlp_out = self.final_mlp(concat_input)
        logits = self.final_fc(mlp_out)
        out = self.get_ctr_output(logits, labels)
        return out


class DCN(BaseModel):
    '''
    DCNv1
    '''
    def __init__(self, args, mode, dataset):
        super(DCN, self).__init__(args, dataset)
        self.deep_arch = args.dcn_deep_arch
        self.cross_net = CrossNet(self.module_inp_dim, args.dcn_cross_num, mode)
        self.deep_net = MLP(self.deep_arch, self.module_inp_dim, self.dropout)
        final_inp_dim = self.module_inp_dim + self.deep_arch[-1]
        self.final_mlp = MLP(self.final_mlp_arch, final_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)


    def forward(self, inp):
        '''
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        '''
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            inp = torch.cat([item_embedding, user_behavior], dim=1)

        deep_part = self.deep_net(inp)
        cross_part = self.cross_net(inp)

        final_inp = torch.cat([deep_part, cross_part], dim=1)
        mlp_out = self.final_mlp(final_inp)
        logits = self.final_fc(mlp_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class DeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(DeepFM, self).__init__(args, dataset)
        # FM
        self.fm_first_iid_emb = nn.Embedding(self.item_num + 1, 1)
        self.fm_first_aid_emb = nn.Embedding(self.attr_num + 1, 1)
        self.fm_first_dense_weight = nn.Parameter(torch.rand([self.dens_vec_num, 1]))
        # DNN
        self.deep_part = MLP(args.deepfm_deep_arch, self.module_inp_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.deepfm_deep_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_vec, labels = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        if self.augment_num:
            dnn_inp = torch.cat([item_embedding, user_behavior, dens_vec], dim=1)
        else:
            dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

        device = next(self.parameters()).device
        # fm first order
        iid_first = self.fm_first_iid_emb(inp['iid'].to(device)).view(-1, 1)
        aid_first = self.fm_first_aid_emb(inp['aid'].to(device)).view(-1, self.attr_fnum)
        linear_sparse_logit = torch.sum(torch.cat([iid_first, aid_first], dim=1), dim=1).view(-1, 1)
        if self.augment_num:
            linear_dense_logit = dens_vec.matmul(self.fm_first_dense_weight).view(-1, 1)
            fm_logit = linear_sparse_logit + linear_dense_logit
        else:
            fm_logit = linear_sparse_logit

        # fm second order
        fm_second_inp = torch.cat([item_embedding, user_behavior], dim=1)
        fm_second_inp = fm_second_inp.view(-1, self.item_fnum + self.hist_fnum, self.embed_dim)

        square_of_sum = torch.pow(torch.sum(fm_second_inp, dim=1, keepdim=True), 2)  # shape: (batch_size,1,embedding_size)
        sum_of_square = torch.sum(torch.pow(fm_second_inp, 2), dim=1, keepdim=True)  # shape: (batch_size,1,embedding_size)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)  # shape: (batch_size,1)
        # print(cross_term.shape)
        fm_logit += cross_term

        # dnn
        deep_out = self.deep_part(dnn_inp)

        logits = fm_logit + self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class xDeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(xDeepFM, self).__init__(args, dataset)
        input_dim = self.field_num * args.embed_dim
        cin_layer_units = args.cin_layer_units
        self.cin = CIN(self.field_num, cin_layer_units)
        self.dnn = MLP(args.final_mlp_arch, input_dim, self.dropout)
        final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
        self.final_fc = nn.Linear(final_dim, args.output_dim)

    def forward(self, inp):
        inp, labels = self.get_filed_input(inp)

        final_vec = self.cin(inp)
        dnn_vec = self.dnn(inp.flatten(start_dim=1))
        final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        logits = self.final_fc(final_vec)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class AutoInt(BaseModel):
    def __init__(self, args, dataset):
        super(AutoInt, self).__init__(args, dataset)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(args.embed_dim if i == 0 else args.num_attn_heads * args.attn_size,
                                     attention_dim=args.attn_size,
                                     num_heads=args.num_attn_heads,
                                     dropout_rate=args.dropout,
                                     use_residual=args.res_conn,
                                     use_scale=args.attn_scale,
                                     layer_norm=False,
                                     align_to='output')
              for i in range(args.num_attn_layers)])
        final_dim = self.field_num * args.attn_size * args.num_attn_heads

        self.attn_out = nn.Linear(final_dim, 1)

    def forward(self, inp):
        inp, labels = self.get_filed_input(inp)
        attention_out = self.self_attention(inp)
        attention_out = torch.flatten(attention_out, start_dim=1)

        logits = self.attn_out(attention_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class FiBiNet(BaseModel):
    def __init__(self, args, dataset):
        super(FiBiNet, self).__init__(args, dataset)
        self.senet_layer = SqueezeExtractionLayer(self.field_num, args.reduction_ratio)
        self.bilinear_layer = BilinearInteractionLayer(self.embed_dim, self.field_num, args.bilinear_type)
        final_dim = self.field_num * (self.field_num - 1) * self.embed_dim
        self.dnn = MLP(args.final_mlp_arch, final_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        feat_embed, labels = self.get_filed_input(inp)
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        comb_out = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)

        logits = self.fc_out(self.dnn(comb_out))
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class FiGNN(BaseModel):
    def __init__(self, args, dataset):
        super(FiGNN, self).__init__(args, dataset)
        self.fignn = FiGNNBlock(self.field_num, self.embed_dim, args.gnn_layer_num,
                                args.res_conn, args.reuse_graph_layer)
        self.fc = AttentionalPrediction(self.field_num, self.embed_dim)

    def forward(self, inp):
        feat_embed, labels = self.get_filed_input(inp)
        h_out = self.fignn(feat_embed)
        logits = self.fc(h_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class DLCM(BaseModel):
    def __init__(self, args, dataset):
        super(DLCM, self).__init__(args, dataset)
        self.gru = torch.nn.GRU(self.module_inp_dim, self.hidden_size, dropout=self.rnn_dp, batch_first=True)
        self.phi_function = Phi_function(self.hidden_size, self.hidden_size, self.dropout)

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        seq_state, final_state = self.gru(processed_inp)
        final_state = torch.squeeze(final_state, dim=0)

        scores = self.phi_function(seq_state, final_state)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class PRM(BaseModel):
    def __init__(self, args, dataset):
        super(PRM, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.pos_embedding = torch.tensor(self.get_pos_embedding(self.max_list_len,
                                                                 self.module_inp_dim)).float().to(args.device)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def get_pos_embedding(self, max_len, d_emb):
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def forward(self, inp):
        processed_inp, labels = self.process_rerank_inp(inp)
        item_embed = processed_inp + self.pos_embedding

        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs


class SetRank(BaseModel):
    def __init__(self, args, dataset):
        super(SetRank, self).__init__(args, dataset)
        self.attention = nn.MultiheadAttention(self.module_inp_dim, args.n_head, batch_first=True,
                                               dropout=args.attn_dp)
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim * 2, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embed, labels = self.process_rerank_inp(inp)
        attn_out, _ = self.attention(item_embed, item_embed, item_embed)
        mlp_out = self.mlp(torch.cat([attn_out, item_embed], dim=-1))
        scores = self.fc_out(mlp_out).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels, attn=True)
        return outputs


class MIR(BaseModel):
    def __init__(self, args, dataset):
        super(MIR, self).__init__(args, dataset)
        self.intra_item_attn = nn.MultiheadAttention(self.itm_emb_dim, args.n_head, batch_first=True,
                                                     dropout=args.attn_dp)
        self.intra_hist_gru = nn.GRU(self.hist_emb_dim, self.hidden_size, dropout=self.rnn_dp,
                                     batch_first=True)
        self.i_fnum = self.item_fnum * 2
        self.h_fnum = self.hist_fnum + self.hidden_size // self.embed_dim

        self.set2list_attn = SLAttention(self.i_fnum, self.h_fnum, self.embed_dim,
                                         self.max_list_len, self.max_hist_len)
        self.module_inp_dim = (self.item_fnum * 3 + self.h_fnum * 2) * self.embed_dim
        self.mlp = MLP(args.final_mlp_arch, self.module_inp_dim, self.dropout)
        self.fc_out = nn.Linear(args.final_mlp_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, dens_vec, orig_dens_list, labels = self.process_input(inp)
        cross_item, _ = self.intra_item_attn(item_embedding, item_embedding, item_embedding)
        cross_hist, _ = self.intra_hist_gru(user_behavior)
        user_seq = torch.cat([user_behavior, cross_hist], dim=-1)
        hist_mean = torch.mean(user_seq, dim=1, keepdim=True)
        hist_mean = hist_mean.repeat([1, self.max_list_len, 1])
        cat_item = torch.cat([item_embedding, cross_item], dim=-1)

        v, q, _, _ = self.set2list_attn(cat_item, user_seq)
        mlp_inp = torch.cat([v, q, item_embedding, hist_mean], dim=-1)
        mlp_out = self.mlp(mlp_inp)
        scores = self.fc_out(mlp_out)
        scores = torch.sigmoid(scores).view(-1, self.max_list_len)
        outputs = self.get_rerank_output(scores, labels)
        return outputs
