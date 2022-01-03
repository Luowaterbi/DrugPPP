import numpy as np

from dgl import DGLGraph
from dgl import readout_nodes
from dgl.nn.pytorch import Set2Set, NNConv, GATConv, AvgPooling

import torch
import torch.nn as nn
import torch.nn.functional as F



class GatherModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True
                           )

    def forward(self, g, n_feat, e_feat, res_connection='all'):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : node features
        """

        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        # print('debug, {}, n feat {}, e_feat {}, out {}'.format(g, n_feat.shape, e_feat.shape, out))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g, out, e_feat))
            else:
                m = torch.relu(self.conv.bias + self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        if res_connection in ['all', 'graph']:
            return out + init
        else:
            return out


class CIGINModel(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 ):
        super(CIGINModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )

        self.fc1 = nn.Linear(8 * self.node_hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.imap = nn.Linear(80, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_solute = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_solvent = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]
        solvent_len = data[3]
        # print("solute.shape=",solute.shape)
        # print("sol")
        # node embeddings after interaction phase
        try:
            # if edge exists in a molecule
            solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            solute_features = self.solute_gather(solute, solute.ndata['x'].float(), None)
        # solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float())
        try:
            # if edge exists in a molecule
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None)

        # print('@@@@@@@@@@@ Debug', solvent_len)
        # Interaction phase
        len_map = torch.mm(solute_len.t(), solvent_len)

        if 'dot' not in self.interaction:
            X1 = solute_features.unsqueeze(0)
            Y1 = solvent_features.unsqueeze(1)
            X2 = X1.repeat(solvent_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, solute_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(solute_features, solvent_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
            interaction_map = torch.mul(len_map.float(), interaction_map)

        solvent_prime = torch.mm(interaction_map.t(), solute_features)
        solute_prime = torch.mm(interaction_map, solvent_features)

        # Prediction phase
        solute_features = torch.cat((solute_features, solute_prime), dim=1)
        solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)

        solute_features = self.set2set_solute(solute, solute_features)
        solvent_features = self.set2set_solvent(solvent, solvent_features)
        # print('DEBUG!!, solute shape', solute_features.shape, 'solvent shape', solvent_features.shape)

        final_features = torch.cat((solute_features, solvent_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map


class SAIGNModel(nn.Module):
    """
    Our model Transformer Interation Graph Network
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='self_att',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 num_tf_layer=3,
                 tf_norm_type='layer_norm',
                 readout_type='set2set',
                 tf_dim=42,
                 tf_nhead=3,
                 tf_mlp_dim=64,
                 tf_dropout=0.1,
                 pred_layer='linear',
                 sep_emb=True,
                 res_connection='all',
                 att_block='none',
                 ):
        super(SAIGNModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.solute_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.solvent_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )

        #  My Staff
        self.res_connection = res_connection
        self.sep_emb = sep_emb
        if sep_emb:
            self.sep_embedder = nn.Embedding(2, node_hidden_dim)

        if tf_norm_type == 'layer_norm':
            tf_norm = nn.LayerNorm(node_hidden_dim)
        else:
            tf_norm = None

        # interaction
        self.tf_nhead= tf_nhead
        single_tf_layer = nn.TransformerEncoderLayer(
            d_model=node_hidden_dim, nhead=tf_nhead, dim_feedforward=tf_mlp_dim, dropout=tf_dropout)
        self.tf_interaction = torch.nn.TransformerEncoder(
            encoder_layer=single_tf_layer, num_layers=num_tf_layer, norm=tf_norm)
        # self.pred_layer = pred_layer
        # if pred_layer == 'linear':
        #     if readout_type == 'set'
        self.att_block = att_block

        # readout
        self.readout_type = readout_type
        if readout_type == 'set2set':
            self.num_step_set2set = num_step_set2_set
            self.num_layer_set2set = num_layer_set2set

            interacted_dim = 2 * self.node_hidden_dim if res_connection in ['all', 'interact'] else self.node_hidden_dim
            self.solute_readout = Set2Set(interacted_dim, self.num_step_set2set, self.num_layer_set2set)
            self.solvent_readout = Set2Set(interacted_dim, self.num_step_set2set, self.num_layer_set2set)

            readout_dim = 8 * self.node_hidden_dim if res_connection in ['all', 'interact'] else 4 * self.node_hidden_dim
            self.fc1 = nn.Linear(readout_dim, 256)  # x 8 = x2 (cat prime) x2 (cat solvent) x2 (set2set)
            # self.fc1 = nn.Linear(2 * self.node_hidden_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)

        else:
            self.solute_readout = AvgPooling()
            self.solvent_readout = AvgPooling()
            readout_dim = 4 * self.node_hidden_dim if res_connection in ['all', 'interact'] else 2 * self.node_hidden_dim
            self.fc1 = nn.Linear(readout_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 1)

    def forward(self, data):
        solute = data[0]
        solvent = data[1]
        solute_len = data[2]  # Elements are staggered, shape: (batch_size, sum_len), eg: [[1, 1, 1, 0], [0, 0, 0, 1]]
        solvent_len = data[3]  # Elements are staggered, shape: (batch_size, sum_len), eg: [[1, 1, 0, 0], [0, 0, 1, 1]]
        batch_size = solute.batch_size

        # node embeddings after interaction phase
        solute_features = self.solute_gather(solute, solute.ndata['x'].float(), solute.edata['w'].float(), self.res_connection)
        try:
            # if edge exists in a molecule
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), solvent.edata['w'].float(), self.res_connection)
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            solvent_features = self.solvent_gather(solvent, solvent.ndata['x'].float(), None, self.res_connection)

        if self.sep_emb:
            solute_features = self.add_sep_emb(solute_features, True)
            solvent_features = self.add_sep_emb(solvent_features, False)

        # Interaction phase
        # solute_shape and solvent_shape (all_node_num in a batch, feature_dim)
        solute_end = solute_features.shape[0]
        solvent_end = solvent_features.shape[0]
        cat_feature = torch.cat((solute_features, solvent_features), 0)

        if self.att_block == 'none':
            cat_mask = self.get_att_mask(solute_len, solvent_len)
        elif self.att_block == 'self':
            cat_mask = self.get_self_att_mask(solute_len, solvent_len)
        else:
            raise ValueError("Invalid value for att_block type")

        # unsqueeze to set batch second, all item are send to single transformer
        interacted_feature = self.tf_interaction(src=cat_feature.unsqueeze(1), mask=cat_mask)

        # split feature
        interacted_feature = interacted_feature.squeeze(1)  # remove batch dim
        solute_prime = torch.narrow(interacted_feature, dim=0, start=0, length=solute_end)
        solvent_prime = torch.narrow(interacted_feature, dim=0, start=solute_end, length=solvent_end)

        # Readout phase
        if self.res_connection in ['all', 'interact']:
            solute_features = torch.cat((solute_features, solute_prime), dim=1)
            solvent_features = torch.cat((solvent_features, solvent_prime), dim=1)
        else:
            if self.interaction != "none":
                solute_features = solute_prime
                solvent_features = solvent_prime

        solute_features = self.solute_readout(solute, solute_features)
        solvent_features = self.solvent_readout(solvent, solvent_features)

        # Prediction phase
        # print('DEBUG!!, solute shape', solute_features.shape, 'solvent shape', solvent_features.shape)
        final_features = torch.cat((solute_features, solvent_features), 1)
        # print('Debug!! final features', final_features.shape, 'batch size', batch_size)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        ret_interaction_map = None
        return predictions, ret_interaction_map

    def add_sep_emb(self, features, first_part=True):
        """

        Parameters
        ----------
        features: (all_nodes_num, f_dim)
        first_part: true to all ones, else zeros

        Returns
        -------

        """
        if first_part:
            sep_ids = torch.ones(1, features.shape[0], dtype=torch.long, device=features.device)
        else:
            sep_ids = torch.zeros(1, features.shape[0], dtype=torch.long, device=features.device)
        sep_embedding = self.sep_embedder(sep_ids).squeeze(0)
        return features + sep_embedding

    def get_att_mask(self, solute_mask, solvent_mask):
        """ mask attention across different sample pairs """
        source = torch.cat((solute_mask, solvent_mask), 1)
        target = torch.cat((solute_mask, solvent_mask), 1)
        ret = torch.bmm(source.unsqueeze(-1), target.unsqueeze(1))  # shape: (batch_size, sum_all_len, sum_all_len)
        ret = torch.sum(ret, dim=0)  # shape: (sum_all_len, sum_all_len), send all pairs to one tf layer
        return (ret * -1 + 1).byte()  # for torch.transformer true are blocked positions

    def get_self_att_mask(self, solute_mask, solvent_mask):
        """ mask attention across different sample pairs and self attention within molecular """
        r_source = torch.cat((solute_mask, torch.zeros(solvent_mask.shape, device=solute_mask.device)), 1)
        r_target = torch.cat((torch.zeros(solute_mask.shape, device=solute_mask.device), solvent_mask), 1)
        right_mask = torch.bmm(r_source.unsqueeze(-1), r_target.unsqueeze(1))

        l_source = torch.cat((torch.zeros(solute_mask.shape, device=solute_mask.device), solvent_mask), 1)
        l_target = torch.cat((solute_mask, torch.zeros(solvent_mask.shape, device=solute_mask.device)), 1)
        left_mask = torch.bmm(l_source.unsqueeze(-1), l_target.unsqueeze(1))

        ret = right_mask + left_mask  # shape: (batch_size, sum_all_len, sum_all_len)
        ret = torch.sum(ret, dim=0)  # shape: (sum_all_len, sum_all_len), send all pairs to one tf layer
        return (ret * -1 + 1).byte()  # for torch.transformer true are blocked positions
