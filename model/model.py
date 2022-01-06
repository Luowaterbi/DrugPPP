# coding: utf-8
"""
Author: Atma, Luowaterbi
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph_transformer import GraphTransformer, make_graph_transformer, Embeddings
from torch.autograd import Variable
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_
from model.New_MoE import MoE


# ====== Model definition ======
def make_model(opt):
    # make encoder
    if opt.encoder == 'gt':  # graph transformer
        encoder_class = GTEncoder if opt.enc_pair_type == 'joint' else PGTEncoder
        encoder = encoder_class(opt,
                                d_features=opt.d_features,
                                n_layer=opt.enc_n_layer,
                                d_model=opt.d_model,
                                n_head=opt.enc_n_head,
                                dropout=opt.enc_dropout,
                                lambda_attention=opt.lambda_attention,
                                lambda_distance=opt.lambda_distance,
                                trainable_lambda=opt.trainable_lambda,
                                n_mlp=2,
                                leaky_relu_slope=0.1,
                                mlp_nonlinear='relu',
                                distance_matrix_kernel='softmax',
                                use_edge_features=opt.use_edge,
                                integrated_dst=opt.integrated_dst,
                                scale_norm=opt.enc_scale_norm,
                                pair_type=opt.enc_pair_type)
    elif opt.encoder == 'mlp':
        encoder = MLPEncoder(opt, d_features=opt.d_features, d_model=opt.d_model, dropout=opt.enc_dropout)
    else:
        raise NotImplementedError('Wrong encoder type: {}'.format(opt.encoder))

    # make interactor
    if opt.interactor == 'sa':
        interactor = SAInteractor(opt,
                                  n_layer=opt.inter_n_layer,
                                  d_model=opt.d_model,
                                  n_head=opt.inter_n_head,
                                  d_mlp=opt.d_model,
                                  dropout=opt.inter_dropout,
                                  norm_type=opt.inter_norm,
                                  type_emb=opt.type_emb,
                                  att_block=opt.att_block,
                                  res_interact=opt.inter_res)
    elif opt.interactor == 'rnsa':
        interactor = RNSAInteractor(opt,
                                    n_layer=opt.inter_n_layer,
                                    d_model=opt.d_model,
                                    n_head=opt.inter_n_head,
                                    d_mlp=opt.d_model,
                                    dropout=opt.inter_dropout,
                                    norm_type=opt.inter_norm,
                                    type_emb=opt.type_emb,
                                    att_block=opt.att_block,
                                    res_interact=opt.inter_res)
    elif opt.interactor == 'rn':
        interactor = RNInteractor(opt)
    elif opt.interactor == 'simple':
        interactor = SimpleInteractor(opt.inter_res)
    else:
        interactor = None

    # make MoE
    # print("MoE=", opt.moe)
    if opt.moe:
        print("built moe")
        Solu_MoE = MoE(input_size=opt.d_model, output_size=opt.d_model, num_experts=opt.num_experts, hidden_size=opt.d_model * 2, noisy_gating=opt.noisy_gating, k=opt.num_used_experts, loss_coef=opt.moe_loss_coef, dropout=opt.moe_dropout)
        Solv_MOE = MoE(input_size=opt.d_model, output_size=opt.d_model, num_experts=opt.num_experts, hidden_size=opt.d_model * 2, noisy_gating=opt.noisy_gating, k=opt.num_used_experts, loss_coef=opt.moe_loss_coef, dropout=opt.moe_dropout)
        if opt.mix:
            print("built mix moe")
            Mix_MoE = MoE(input_size=opt.d_model, output_size=opt.d_model, num_experts=opt.num_experts, hidden_size=opt.d_model * 2, noisy_gating=opt.noisy_gating, k=opt.num_used_experts, loss_coef=opt.moe_loss_coef, dropout=opt.moe_dropout)
        else:
            Mix_MoE = None
    else:
        Solu_MoE = None
        Solv_MOE = None
        Mix_MoE = None

    # make readout
    if interactor and opt.inter_res == 'cat' and opt.interactor not in ['rn']:
        dec_d_model = 2 * opt.d_model  # do res connection, and cat interacted and raw encoded feature
    else:
        dec_d_model = opt.d_model
    if opt.readout in ['rn', 'j_avg']:
        dec_input = dec_d_model  # relation node feature
    elif opt.readout in ['rn_avg', 'rn_sum']:
        dec_input = dec_d_model * 3  # joint feature is cat of 'two' input and relation node feature
    else:
        dec_input = dec_d_model * 2  # joint feature is cat of 'two' input
    readout_layer = ReadoutLayer(dec_d_model, mix_moe=opt.moe & opt.mix, readout=opt.readout)

    # make decoder
    if opt.decoder == 'reg':
        decoder = RegressionDecoder(d_input=dec_input, readout=readout_layer)
    elif opt.decoder == 'cls':
        decoder = ClassificationDecoder(d_input=dec_input, readout=readout_layer, num_tags=opt.num_tags)
    else:
        raise NotImplementedError('Wrong decoder type: {}'.format(opt.decoder))

    model = SAIGNModel(opt, encoder, interactor, Solu_MoE, Solv_MOE, Mix_MoE, decoder)
    # print(model)
    # Initialization
    # This was important from MAT and older code. e.g Initialize parameters with Glorot / fan_avg.
    # print(model)
    init_model(opt, model)
    return model


# ====== Tools ======
def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return _no_grad_uniform_(tensor, -a, a)


def init_model(opt, model):
    init_type = opt.init_type
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'uniform':
                nn.init.xavier_uniform_(p)
            elif init_type == 'normal':
                nn.init.xavier_normal_(p)
            elif init_type == 'small_normal_init':
                xavier_normal_small_init_(p)
            elif init_type == 'small_uniform_init':
                xavier_uniform_small_init_(p)


# ====== Modules ======
class GTEncoder(nn.Module):
    """
    A single Graph Transformer Encoder (GT), for joint input.
    f: joint_x1_x2 -> x1_x2_feature
    """

    def __init__(self,
                 opt,
                 d_features,
                 n_layer=2,
                 d_model=128,
                 n_head=8,
                 dropout=0.1,
                 lambda_attention=0.3,
                 lambda_distance=0.3,
                 trainable_lambda=False,
                 n_mlp=2,
                 leaky_relu_slope=0.0,
                 mlp_nonlinear='relu',
                 distance_matrix_kernel='softmax',
                 use_edge_features=False,
                 integrated_dst=False,
                 scale_norm=False,
                 pair_type='share'):
        super(GTEncoder, self).__init__()
        self.opt = opt

        self.encoder = make_graph_transformer(d_atom=d_features,
                                              N=n_layer,
                                              d_model=d_model,
                                              h=n_head,
                                              dropout=dropout,
                                              lambda_attention=lambda_attention,
                                              lambda_distance=lambda_distance,
                                              trainable_lambda=trainable_lambda,
                                              N_dense=n_mlp,
                                              leaky_relu_slope=leaky_relu_slope,
                                              dense_output_nonlinearity=mlp_nonlinear,
                                              distance_matrix_kernel=distance_matrix_kernel,
                                              use_edge_features=use_edge_features,
                                              integrated_distances=integrated_dst,
                                              scale_norm=scale_norm)

    def forward(self, batch):
        node_emb, mask, adjacency, distance = batch
        features = self.encoder.encode(src=node_emb, src_mask=mask, adj_matrix=adjacency, distances_matrix=distance, edges_att=None)
        return features, None, mask, None


class PGTEncoder(nn.Module):
    """
    A Pair-wise Graph Transformer (PGT) encoder that consists of two separated / one shared Graph Transformer Encoder.
    f: (x1, x2) -> (x1_feature, x2_feature)
    """

    def __init__(self,
                 opt,
                 d_features,
                 n_layer=2,
                 d_model=128,
                 n_head=8,
                 dropout=0.1,
                 lambda_attention=0.3,
                 lambda_distance=0.3,
                 trainable_lambda=False,
                 n_mlp=2,
                 leaky_relu_slope=0.0,
                 mlp_nonlinear='relu',
                 distance_matrix_kernel='softmax',
                 use_edge_features=False,
                 integrated_dst=False,
                 scale_norm=False,
                 pair_type='share'):
        super(PGTEncoder, self).__init__()
        self.opt = opt

        self.x1_encoder = make_graph_transformer(d_atom=d_features,
                                                 N=n_layer,
                                                 d_model=d_model,
                                                 h=n_head,
                                                 dropout=dropout,
                                                 lambda_attention=lambda_attention,
                                                 lambda_distance=lambda_distance,
                                                 trainable_lambda=trainable_lambda,
                                                 N_dense=n_mlp,
                                                 leaky_relu_slope=leaky_relu_slope,
                                                 dense_output_nonlinearity=mlp_nonlinear,
                                                 distance_matrix_kernel=distance_matrix_kernel,
                                                 use_edge_features=use_edge_features,
                                                 integrated_distances=integrated_dst,
                                                 scale_norm=scale_norm)

        if pair_type == 'share':
            self.x2_encoder = self.x1_encoder
        else:
            self.x2_encoder = make_graph_transformer(d_atom=d_features,
                                                     N=n_layer,
                                                     d_model=d_model,
                                                     h=n_head,
                                                     dropout=dropout,
                                                     lambda_attention=lambda_attention,
                                                     lambda_distance=lambda_distance,
                                                     trainable_lambda=trainable_lambda,
                                                     N_dense=n_mlp,
                                                     leaky_relu_slope=leaky_relu_slope,
                                                     dense_output_nonlinearity=mlp_nonlinear,
                                                     distance_matrix_kernel=distance_matrix_kernel,
                                                     use_edge_features=use_edge_features,
                                                     integrated_distances=integrated_dst,
                                                     scale_norm=scale_norm)

    def forward(self, batch):
        node_emb1, mask1, adjacency1, distance1, node_emb2, mask2, adjacency2, distance2 = batch
        x1_features = self.x1_encoder.encode(src=node_emb1, src_mask=mask1, adj_matrix=adjacency1, distances_matrix=distance1, edges_att=None)
        x2_features = self.x2_encoder.encode(src=node_emb2, src_mask=mask2, adj_matrix=adjacency2, distances_matrix=distance2, edges_att=None)
        return x1_features, x2_features, mask1, mask2


class MLPEncoder(nn.Module):
    """
    A pair-wise encoder that consists of two separated / one shared Graph Transformer Encoder.
    f: (x1, x2) -> (x1_feature, x2_feature)
    """

    def __init__(self, opt, d_features, d_model=128, dropout=0.1, pair_type='share'):
        super(MLPEncoder, self).__init__()
        self.opt = opt
        self.x1_encoder = nn.Sequential(
            Embeddings(d_model=d_model, d_atom=d_features, dropout=dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        if pair_type == 'share':
            self.x2_encoder = self.x1_encoder
        else:
            self.x2_encoder = nn.Sequential(
                Embeddings(d_model=d_model, d_atom=d_features, dropout=dropout),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

    def forward(self, batch):
        node_emb1, mask1, adjacency1, distance1, node_emb2, mask2, adjacency2, distance2 = batch
        x1_features = self.x1_encoder(node_emb1)
        x2_features = self.x2_encoder(node_emb2)
        return x1_features, x2_features, mask1, mask2


class RNInteractor(nn.Module):
    """
        A simple interaction layer that extract relation node feature as relation features:
            f: joint_features -> (x1', x2', I),
        where x1', x2' are interacted features and I is relation features (relation_node_features).
        """

    def __init__(self, opt, *args, **kwargs):
        super(RNInteractor, self).__init__()

    def forward(self, x1, x2, x1_mask, x2_mask):
        """
        Parameters
        ----------
        x1: shape [batch x node_num x emb_dim) joint features of x1, x2, relation node
        x2: None
        x1_mask: shape [batch x node_num],  joint mask, true/1 are valid positions, false/0 are masked position
        x2_mask: None

        Returns
        -------
        (x1', x2', I), where x1', x2' are interacted features and I is relation features (relation_node_features).
        """
        # todo: replace None in returns with extracted x1, x2 features
        relation_features = torch.narrow(x1, dim=1, start=0, length=1).squeeze(1)  # shape [batch_size x emb_dim]
        return x1, None, relation_features, x1, x1_mask


# class CIGINInteractor(nn.Module):
#     def __init__(self, ):


class SAInteractor(nn.Module):
    """
    Interaction layer with transformer:
    f: (x1, x2) -> (x1', x2', I),
    where x1', x2' are interacted features and I is relation features.
    """

    def __init__(self, opt, n_layer=2, d_model=128, n_head=4, d_mlp=128, dropout=0.1, norm_type='layer_norm', type_emb='sep', att_block='none', res_interact='none'):
        super(SAInteractor, self).__init__()

        self.att_block = att_block
        self.type_emb = type_emb
        self.res_interact = res_interact

        if type_emb == 'sep':
            self.type_embedder = nn.Embedding(2, d_model)
        else:
            self.type_embedder = None

        if norm_type == 'layer_norm':
            norm = nn.LayerNorm(d_model)
        else:
            norm = None

        single_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_mlp, dropout=dropout)
        self.interaction = torch.nn.TransformerEncoder(encoder_layer=single_layer, num_layers=n_layer, norm=norm)

    def forward(self, x1, x2, x1_mask, x2_mask):
        """
        Parameters
        ----------
        x1: shape [batch x node_num x emb_dim)
        x2: shape [batch x node_num x emb_dim)
        x1_mask: shape [batch x node_num],  true/1 are valid positions, false/0 are masked position
        x2_mask: shape [batch x node_num],  true/1 are valid positions, false/0 are masked position

        Returns
        -------
        (x1', x2', I), where x1', x2' are interacted features and I is relation features.
        """
        x1_end, x2_end = x1.shape[1], x2.shape[1]

        if self.type_emb == 'sep':
            x1_input, x2_input = self.add_sep_emb(x1, True), self.add_sep_emb(x2, False)
        else:
            x1_input, x2_input = x1, x2

        cat_feature = torch.cat((x1_input, x2_input), 1)  # [batch x node_num x emb_dim], find the position
        cat_mask = torch.cat((x1_mask, x2_mask), 1)  # [batch x node_num]
        cat_mask = ~cat_mask  # for torch.transformer true are masked positions, ~ to reverse it

        # interaction phase, permute to make batch second
        interacted_feature = self.interaction(src=cat_feature.permute(1, 0, 2), src_key_padding_mask=cat_mask)
        relation_features = None

        # split feature
        interacted_feature = interacted_feature.permute(1, 0, 2)
        x1_prime = torch.narrow(interacted_feature, dim=1, start=0, length=x1_end)
        x2_prime = torch.narrow(interacted_feature, dim=1, start=x1_end, length=x2_end)
        if self.res_interact == 'cat':
            x1_prime = torch.cat((x1, x1_prime), dim=-1)
            x2_prime = torch.cat((x2, x2_prime), dim=-1)
        return x1_prime, x2_prime, relation_features, interacted_feature, ~cat_mask

    def add_sep_emb(self, features, first_part=True):
        """
        Parameters
        ----------
        features: shape [batch x nodes_num x f_dim]
        first_part: true to use type1, else type2

        Returns
        -------
        shape [batch x nodes_num x f_dim]
        """
        if first_part:
            sep_ids = torch.ones(features.shape[0], features.shape[1], dtype=torch.long, device=features.device)
        else:
            sep_ids = torch.zeros(features.shape[0], features.shape[1], dtype=torch.long, device=features.device)
        return features + self.type_embedder(sep_ids)


class RNSAInteractor(SAInteractor):
    """ Self-attentive interactor with relation node """

    def __init__(self, opt, n_layer=2, d_model=128, n_head=4, d_mlp=128, dropout=0.1, norm_type='layer_norm', type_emb='sep', att_block='none', res_interact='none'):
        super(RNSAInteractor, self).__init__(opt, n_layer, d_model, n_head, d_mlp, dropout, norm_type, type_emb, att_block, res_interact)
        self.relation_emb = nn.Parameter(torch.randn(d_model, dtype=torch.float), requires_grad=True)

    def forward(self, x1, x2, x1_mask, x2_mask):
        """
        Parameters
        ----------
        x1: shape [batch x node_num x emb_dim)
        x2: shape [batch x node_num x emb_dim)
        x1_mask: shape [batch x node_num],  true/1 are valid positions, false/0 are masked position
        x2_mask: shape [batch x node_num],  true/1 are valid positions, false/0 are masked position

        Returns
        -------
        (x1', x2', I), where x1', x2' are interacted features and I is relation features.
        """
        x1_end, x2_end = x1.shape[1], x2.shape[1]

        if self.type_emb == 'sep':
            x1_input, x2_input = self.add_sep_emb(x1, True), self.add_sep_emb(x2, False)
        else:
            x1_input, x2_input = x1, x2

        rel_feature = self.relation_emb.expand(x1.shape[0], 1, x1.shape[-1]).to(x1.device)  # [batch x 1 x emb_dim]
        rel_mask = torch.ones(x1.shape[0], 1, dtype=torch.bool).to(x1.device)  # [batch x 1]

        cat_feature = torch.cat((rel_feature, x1_input, x2_input), 1)  # [batch x node_num+1 x emb_dim]
        cat_mask = torch.cat((rel_mask, x1_mask, x2_mask), 1)  # [batch x node_num+1]
        cat_mask = ~cat_mask  # for torch.transformer true are masked positions, ~ to reverse it

        # interaction phase, permute to make batch second
        interacted_feature = self.interaction(src=cat_feature.permute(1, 0, 2), src_key_padding_mask=cat_mask)

        # split feature
        interacted_feature = interacted_feature.permute(1, 0, 2)
        relation_features = torch.narrow(interacted_feature, dim=1, start=0, length=1).squeeze(1)  # shape [batch_size x emb_dim]
        x1_prime = torch.narrow(interacted_feature, dim=1, start=1, length=x1_end)
        x2_prime = torch.narrow(interacted_feature, dim=1, start=x1_end + 1, length=x2_end)
        if self.res_interact == 'cat':
            x1_prime = torch.cat((x1, x1_prime), dim=-1)
            x2_prime = torch.cat((x2, x2_prime), dim=-1)
        elif self.res_interact == 'no_inter':
            x1_prime = x1
            x2_prime = x2
        return x1_prime, x2_prime, relation_features, interacted_feature, ~cat_mask


class SimpleInteractor(nn.Module):

    def __init__(self, res_interact):
        super(SimpleInteractor, self).__init__()
        self.res_interact = res_interact

    def forward(self, x1, x2, x1_mask, x2_mask):
        x1_prime = block_padding(x1, x1_mask)
        x2_prime = block_padding(x2, x2_mask)
        interaction_map = torch.einsum("bnd,bmd->bnm", x1_prime, x2_prime)
        interaction_map = torch.tanh(interaction_map)
        x1_prime = x1_prime + torch.einsum("bnm,bmd->bnd", interaction_map, x2_prime)
        x2_prime = x2_prime + torch.einsum("bnm,bnd->bmd", interaction_map, x1_prime)
        relation_feature = None
        cat_feature = torch.cat([x1_prime, x2_prime], dim=1)
        cat_mask = torch.cat([x1_mask, x2_mask], dim=1)
        if self.res_interact == 'cat':
            x1_prime = torch.cat((x1, x1_prime), dim=-1)
            x2_prime = torch.cat((x2, x2_prime), dim=-1)
        elif self.res_interact == 'no_inter':
            x1_prime = x1
            x2_prime = x2
        return x1_prime, x2_prime, relation_feature, cat_feature, cat_mask


class Set2Set(nn.Module):

    def __init__(self, input_dim, hidden_dim, act_fn=nn.ReLU, num_layers=1):
        '''
        Args:
            input_dim: input dim of Set2Set.
            hidden_dim: the dim of set representation, which is also the INPUT dimension of
                the LSTM in Set2Set.
                This is a concatenation of weighted sum of embedding (dim input_dim), and the LSTM
                hidden/output (dim: self.lstm_output_dim).
        '''
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if hidden_dim <= input_dim:
            print('ERROR: Set2Set output_dim should be larger than input_dim')
        # the hidden is a concatenation of weighted sum of embedding and LSTM output
        self.lstm_output_dim = hidden_dim - input_dim
        self.lstm = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)

        # convert back to dim of input_dim
        self.pred = nn.Linear(hidden_dim, input_dim)
        self.act = act_fn()

    def forward(self, embedding, mask):
        '''
        Args:
            embedding: [batch_size x n x d] embedding matrix
            mask: [batch_size x n] True/1 for padding pos
        Returns:
            aggregated: [batch_size x d] vector representation of all embeddings
        '''
        batch_size = embedding.size()[0]
        n = embedding.size()[1]

        hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).cuda(), torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).cuda())

        q_star = torch.zeros(batch_size, 1, self.hidden_dim).cuda()
        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, hidden = self.lstm(q_star, hidden)
            # e: batch_size x n x 1
            e = embedding @ torch.transpose(q, 1, 2)
            a = nn.Softmax(dim=1)(e)
            r = torch.sum(a * embedding, dim=1, keepdim=True)
            q_star = torch.cat((q, r), dim=2)
        q_star = torch.squeeze(q_star, dim=1)
        out = self.act(self.pred(q_star))

        return out


def block_padding(features, mask):
    """ Set padding features to be zero """
    mask = mask.long().unsqueeze(-1).expand(mask.shape[0], mask.shape[1], features.shape[-1])
    return torch.mul(features, mask)


def masked_mean(features, mask, dim):
    return block_padding(features, mask).sum(dim) / mask.long().sum(-1).unsqueeze(-1)


def masked_sum(features, mask, dim):
    return block_padding(features, mask).sum(dim)


class ReadoutLayer(nn.Module):
    """ readout pair-wise features """

    def __init__(self, d_input, mix_moe=False, readout='avg'):
        super(ReadoutLayer, self).__init__()
        self.readout = readout
        self.mix_moe = mix_moe
        if readout == 'set2set':
            self.x1_readout = Set2Set(d_input, d_input * 2)
            self.x2_readout = Set2Set(d_input, d_input * 2)
            # raise NotImplementedError("mask padding for set2set has not been added.")
        elif readout == 'shared_set2set':
            self.x1_readout = Set2Set(d_input, d_input)
            self.x2_readout = self.x1_readout
            raise NotImplementedError("mask padding for set2set has not been added.")

    def forward(self, x1, x2, mask1, mask2, relation_features=None, relation_mask=None):
        """

        Parameters
        ----------
        x1: input features [batch x nodes_num x f_dim]
        x2: input features [batch x nodes_num x f_dim]
        mask1: [batch x nodes_num], True for valid pos
        mask2: [batch x nodes_num], True for valid pos
        relation_features: features for x1, x2 relation

        Returns
        -------

        """
        if self.readout == 'avg':
            x1, x2 = masked_mean(x1, mask1, dim=1), masked_mean(x2, mask2, dim=1)
            joint_feature = torch.cat((x1, x2), -1)
        elif self.readout == 'set2set':
            x1, x2 = self.x1_readout(x1, mask1), self.x2_readout(x2, mask2)
            joint_feature = torch.cat((x1, x2), -1)
        elif self.readout == 'rn':
            if self.mix_moe:
                relation_features = torch.narrow(relation_features, dim=1, start=0, length=1).squeeze(1)  # shape [batch_size x emb_dim]
            joint_feature = relation_features
        elif self.readout == 'rn_avg':
            if x2 is None:
                node_feature = masked_mean(x1[:, 1:, :], mask1[:, 1:], dim=1)
                raise NotImplementedError("Need to cat it with max pooling res")
            else:
                x1, x2 = masked_mean(x1, mask1, dim=1), masked_mean(x2, mask2, dim=1)
                node_feature = torch.cat((x1, x2), -1)
            if self.mix_moe:
                relation_features = masked_mean(relation_features, relation_mask, dim=1)
            joint_feature = torch.cat((relation_features, node_feature), -1)
            # print('debug joint feature',  joint_feature.shape)
        elif self.readout == 'rn_sum':
            if x2 is None:
                node_feature = masked_sum(x1[:, 1:, :], mask1[:, 1:], dim=1)
                raise NotImplementedError("Need to cat it with max pooling res")
            else:
                x1, x2 = masked_sum(x1, mask1, dim=1), masked_sum(x2, mask2, dim=1)
                node_feature = torch.cat((x1, x2), -1)
            if self.mix_moe:
                relation_features = masked_sum(relation_features, relation_mask, dim=1)
            joint_feature = torch.cat((relation_features, node_feature), -1)
        elif self.readout == 'j_avg':
            joint_feature = masked_mean(x1[:, 1:, :], mask1[:, 1:], dim=1)
        else:
            raise ValueError('Wrong readout type: {}'.format(self.readout))

        return joint_feature


class RegressionDecoder(nn.Module):
    """ readout feature and make predictions """

    def __init__(self, d_input, readout):
        super(RegressionDecoder, self).__init__()
        self.readout = readout
        self.map_layer = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.Tanh(),
            nn.Linear(d_input, 1),
        )

    def forward(self, x1, x2, mask1, mask2, moe_loss, relation_features=None, relation_mask=None):
        """

        Parameters
        ----------
        x1: input features [batch x nodes_num x f_dim]
        x2: input features [batch x nodes_num x f_dim]
        mask1: [batch x nodes_num], True for valid pos
        mask2: [batch x nodes_num], True for valid pos
        relation_features: useless for now

        Returns
        -------

        """
        joint_feature = self.readout(x1, x2, mask1, mask2, relation_features, relation_mask)
        pred = self.map_layer(joint_feature).squeeze(-1)
        return pred, moe_loss


class ClassificationDecoder(nn.Module):
    """ readout feature and make predictions """

    def __init__(self, d_input, readout, num_tags):
        super(ClassificationDecoder, self).__init__()
        self.readout = readout
        self.map_layer = nn.Sequential(nn.Linear(d_input, d_input), nn.ReLU(), nn.Linear(d_input, num_tags), nn.Softmax(dim=-1))

    def forward(self, x1, x2, mask1, mask2, moe_loss, relation_features=None, relation_mask=None):
        """

        Parameters
        ----------
        x1: input features [batch x nodes_num x f_dim]
        x2: input features [batch x nodes_num x f_dim]
        mask1: [batch x nodes_num], True for valid pos
        mask2: [batch x nodes_num], True for valid pos
        relation_features: useless for now

        Returns
        -------

        """
        joint_feature = self.readout(x1, x2, mask1, mask2, relation_features, relation_mask)
        pred = self.map_layer(joint_feature)
        return pred, moe_loss


def moe_input_process(moe_input, features, mask):
    if moe_input == "mol_avg":
        features = masked_mean(features, mask, dim=1).unsqueeze(1)
        mask = (torch.ones(mask.shape[0], 1) > 0).to(mask.device)
    elif moe_input == "mol_sum":
        features = masked_sum(features, mask, dim=1).unsqueeze(1)
        mask = (torch.ones(mask.shape[0], 1) > 0).to(mask.device)
    return features, mask


# ====== full models ======
class SAIGNModel(nn.Module):
    """
    Self Attentive Interaction Graph Network for pair-wise prediction: f: (x1, x2) -> y
    """

    def __init__(self, opt, encoder, interactor, Solu_MoE, Solv_MoE, Mix_MoE, decoder):
        super(SAIGNModel, self).__init__()
        self.opt = opt
        self.encoder = encoder
        self.interactor = interactor
        self.decoder = decoder
        self.Solv_MoE = Solv_MoE
        self.Solu_MoE = Solu_MoE
        self.Mix_MoE = Mix_MoE

    def forward(self, batch, _print=False):
        """

        Parameters
        ----------
        batch: a list of model input, including x1 and x2 features, (embeddings, masks, adjacency_mat, distance_mat,)
            and y as output label.

        Returns
        -------
        model prediction of current task.
        """
        x1_features, x2_features, mask1, mask2 = self.encoder(batch)

        cat_mask = None
        if self.interactor:
            x1_features, x2_features, relation_features, cat_features, cat_mask = self.interactor(x1_features, x2_features, mask1, mask2)
            if self.Mix_MoE:
                # mol_sum/mol_avg 并不会只用relation node，而是整个cat features的avg或者sum
                cat_features, cat_mask = moe_input_process(self.opt.moe_input, cat_features, cat_mask)
                if _print:
                    print("Mix_MoE:")
                relation_features, rn_moe_loss = self.Mix_MoE(cat_features, cat_mask, train=self.training, _print=_print)
            else:
                rn_moe_loss = 0
        else:
            relation_features = None

        if self.opt.moe:
            x1_moe_features, mask1 = moe_input_process(self.opt.moe_input, x1_features, mask1)
            x2_moe_features, mask2 = moe_input_process(self.opt.moe_input, x2_features, mask2)
            if _print:
                print('Solu_MoE:')
            x1_features, x1_moe_loss = self.Solu_MoE(x1_moe_features, mask1, train=self.training, _print=_print)
            if _print:
                print('Solv_MoE:')
            x2_features, x2_moe_loss = self.Solv_MoE(x2_moe_features, mask2, train=self.training, _print=_print)
        else:
            x1_moe_loss = 0
            x2_moe_loss = 0

        moe_loss = x1_moe_loss + x2_moe_loss + rn_moe_loss
        return self.decoder(x1_features, x2_features, mask1, mask2, moe_loss, relation_features, cat_mask)
