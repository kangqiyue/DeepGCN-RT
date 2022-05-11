import os
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch import GATConv
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn.gat import  GATLayer
from dgllife.model.gnn.gcn import GCNLayer
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout


class GINLayerModified(nn.Module):
    def __init__(self, num_edge_emb, emb_dim, batch_norm=True, activation=None):
        super(GINLayerModified, self).__init__()
        self.edge_embeddings = nn.Linear(num_edge_emb, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        self.edge_embeddings.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        edge_embeds = self.edge_embeddings(edge_feats)
        g.ndata['feat'] = node_feats
        g.edata['feat'] = edge_embeds
        g.update_all(fn.u_add_e('feat', 'feat', 'm'), fn.sum('m', 'feat'))

        node_feats = self.mlp(g.ndata.pop('feat'))
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats


# class EmbeddingLayer(nn.Module):
#     def __init__(self, node_emb_dim, edge_emb_dim=None):
#         super(EmbeddingLayer, self).__init__()
#         self.node_emb_dim= node_emb_dim
#         self.edge_emb_dim=edge_emb_dim
#
#         self.atom_encoder = AtomEncoder(node_emb_dim)
#         self.float_node_embeddings = nn.Linear(6, node_emb_dim, bias=False)
#         if edge_emb_dim is not None:
#             self.bond_encoder = BondEncoder(edge_emb_dim)
#
#     def forward(self, g):
#         node_feats, edge_feats, node_float_feats = g.ndata["node_feat"], g.edata["edge_feat"], g.ndata["node_float_feat"]
#         node_feats = self.atom_encoder(node_feats)
#         node_feats += self.float_node_embeddings(node_float_feats)
#
#         if self.edge_emb_dim is None:
#             return node_feats
#         else:
#             edge_feats = self.bond_encoder(edge_feats)
#             return  node_feats, edge_feats


class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)

    def forward(self, g):
        node_feats, edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats


'''GAT model'''
# from dgllife.model.gnn.gat import GAT
# Code was adapted and modified from DGL-LifeSci, url: https://lifesci.dgl.ai/api/model.gnn.html
class GATModel(nn.Module):
    """GAT from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 1 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        all-head results for each GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, no activation is applied for each GAT layer.
    biases : list of bool
        ``biases[i]`` gives whether to use bias for the i-th GAT layer. ``len(activations)``
        equals the number of GAT layers. By default, we use bias for all GAT layers.
    """
    def __init__(self, node_in_dim, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None):
        super(GATModel, self).__init__()
        if hidden_feats is None:
            hidden_feats = [200, 200, 200,200,200]
        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0])

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [1 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.1 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.1 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(num_heads), len(feat_drops), len(attn_drops),
                   len(alphas), len(residuals), len(agg_modes), len(activations), len(biases)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, num_heads, ' \
                                       'feat_drops, attn_drops, alphas, residuals, ' \
                                       'agg_modes, activations, and biases to be the same, ' \
                                       'got {}'.format(lengths)
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()

        in_feats = hidden_feats[0]
        for i in range(n_layers):
            self.gnn_layers.append(GATLayer(in_feats, hidden_feats[i], num_heads[i],
                                            feat_drops[i], attn_drops[i], alphas[i],
                                            residuals[i], agg_modes[i], activations[i],
                                            biases[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        feats = self.embed_layer(g)
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        # add mlp layers
        g.ndata['feats'] = feats
        feats = dgl.sum_nodes(g, "feats")
        feats = self.out(feats)

        return feats


'''GCN model'''
# Code was adapted and modified from DGL-LifeSci, url: https://lifesci.dgl.ai/api/model.gnn.html
class GCNModel(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    """

    def __init__(self, node_in_dim, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCNModel, self).__init__()

        if hidden_feats is None:
            hidden_feats = [200, 200, 200,200,200]
        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0])

        in_feats = hidden_feats[0]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.1 for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(gnn_norm), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

        # mlp layers
        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        feats = self.embed_layer(g)
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        g.ndata['feats'] = feats
        feats = dgl.sum_nodes(g, "feats")
        feats = self.out(feats)

        return feats


'''GCN model with attention and GRU readout'''
# Code was adapted and modified from DGL-LifeSci, url: https://lifesci.dgl.ai/api/model.gnn.html
class GCNModelAFPreadout(nn.Module):
    r"""GCN from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the size of node representations after the i-th GCN layer.
        ``len(hidden_feats)`` equals the number of GCN layers.  By default, we use
        ``[64, 64]``.
    gnn_norm : list of str
        ``gnn_norm[i]`` gives the message passing normalizer for the i-th GCN layer, which
        can be `'right'`, `'both'` or `'none'`. The `'right'` normalizer divides the aggregated
        messages by each node's in-degree. The `'both'` normalizer corresponds to the symmetric
        adjacency normalization in the original GCN paper. The `'none'` normalizer simply sums
        the messages. ``len(gnn_norm)`` equals the number of GCN layers. By default, we use
        ``['none', 'none']``.
    activation : list of activation functions or None
        If not None, ``activation[i]`` gives the activation function to be used for
        the i-th GCN layer. ``len(activation)`` equals the number of GCN layers.
        By default, ReLU is applied for all GCN layers.
    residual : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GCN layer.
        ``len(residual)`` equals the number of GCN layers. By default, residual connection
        is performed for each GCN layer.
    batchnorm : list of bool
        ``batchnorm[i]`` decides if batch normalization is to be applied on the output of
        the i-th GCN layer. ``len(batchnorm)`` equals the number of GCN layers. By default,
        batch normalization is applied for all GCN layers.
    dropout : list of float
        ``dropout[i]`` decides the dropout probability on the output of the i-th GCN layer.
        ``len(dropout)`` equals the number of GCN layers. By default, no dropout is
        performed for all layers.
    """

    def __init__(self, node_in_dim, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCNModelAFPreadout, self).__init__()

        if hidden_feats is None:
            hidden_feats = [200, 200, 200,200,200]
        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0])

        in_feats = hidden_feats[0]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0.1 for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(gnn_norm), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

        self.readout = AttentiveFPReadout(
            hidden_feats[-1], num_timesteps=2, dropout=dropout[-1]
        )

        # mlp layers
        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization

        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] in initialization.
        """
        feats = self.embed_layer(g)
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        # g.ndata['feats'] = feats
        # feats = dgl.sum_nodes(g, "feats")
        # feats = self.out(feats)
        out = self.readout(g, feats)
        out = self.out(out)
        return out

'''GIN model'''
# from dgllife.model.gnn.gin import GIN
class GINModel(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__

    This module is for updating node representations only.

    Parameters
    ----------
    num_node_emb_list : list of int
        num_node_emb_list[i] gives the number of items to embed for the
        i-th categorical node feature variables. E.g. num_node_emb_list[0] can be
        the number of atom types and num_node_emb_list[1] can be the number of
        atom chirality types.
    num_edge_emb_list : list of int
        num_edge_emb_list[i] gives the number of items to embed for the
        i-th categorical edge feature variables. E.g. num_edge_emb_list[0] can be
        the number of bond types and num_edge_emb_list[1] can be the number of
        bond direction types.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 200.
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.

        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0
    """
    def __init__(self, num_node_emb, num_edge_emb,
                 num_layers=5, emb_dim=200, JK='last', dropout=0.1):
        super(GINModel, self).__init__()

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.node_embeddings = nn.Linear(num_node_emb, emb_dim)

        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayerModified(num_edge_emb, emb_dim))
            else:
                self.gnn_layers.append(GINLayerModified(num_edge_emb, emb_dim, activation=F.relu))

        self.out = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_embeddings.reset_parameters()
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g):
        """Update node representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        categorical_node_feats : list of LongTensor of shape (N)
            * Input categorical node features
            * len(categorical_node_feats) should be the same as len(self.node_embeddings)
            * N is the total number of nodes in the batch of graphs
        categorical_edge_feats : list of LongTensor of shape (E)
            * Input categorical edge features
            * len(categorical_edge_feats) should be the same as
              len(num_edge_emb_list) in the arguments
            * E is the total number of edges in the batch of graphs

        Returns
        -------
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size. In particular, M will be emb_dim * (num_layers + 1)
            if self.JK == 'concat' and emb_dim otherwise.
        """
        categorical_node_feats, categorical_edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]

        node_embeds = self.node_embeddings(categorical_node_feats)

        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_feats = self.gnn_layers[layer](g, all_layer_node_feats[layer],
                                                categorical_edge_feats)
            node_feats = self.dropout(node_feats)
            all_layer_node_feats.append(node_feats)

        if self.JK == 'concat':
            final_node_feats = torch.cat(all_layer_node_feats, dim=1)
        elif self.JK == 'last':
            final_node_feats = all_layer_node_feats[-1]
        elif self.JK == 'max':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.max(torch.cat(all_layer_node_feats, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            all_layer_node_feats = [h.unsqueeze(0) for h in all_layer_node_feats]
            final_node_feats = torch.sum(torch.cat(all_layer_node_feats, dim=0), dim=0)
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))

        g.ndata['feats'] = final_node_feats
        feats = dgl.sum_nodes(g, "feats")
        feats = self.out(feats)
        return feats


'''AttentiveFP model'''
class AttentivfFPModel(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 num_layers=5,
                 graph_feat_size=200,
                 dropout=0.1,
                 num_timesteps=2,
                 ):
        super(AttentivfFPModel, self).__init__()
        # self.embed_layer = EmbeddingLayerConcat(node_in_dim=node_in_dim, node_emb_dim=node_feat_size,
        #                                         edge_in_dim=edge_in_dim, edge_emb_dim=edge_feat_size)
        self.afp = AttentiveFPGNN(
            node_in_dim,
            edge_in_dim,
            num_layers,
            graph_feat_size,
            dropout,
        )
        self.readout = AttentiveFPReadout(
            graph_feat_size, num_timesteps = num_timesteps, dropout = dropout
        )
        self.out = nn.Sequential(
            nn.Linear(graph_feat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, g):
        # node_feats, edge_feats = self.embed_layer(g)
        node_feats, edge_feats= g.ndata["node_feat"], g.edata["edge_feat"]
        node_feats = self.afp(g, node_feats, edge_feats)
        node_feats = self.readout(g, node_feats)
        out = self.out(node_feats)
        return out


'''deepgcn model'''
#adopted and modified from https://github.com/xnuohz/DeeperGCN-dgl
from modules import norm_layer
from layers import GENConv
class DeeperGCN(nn.Module):
    r"""

    Description
    -----------
    Introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    node_feat_dim: int
        Size of node feature dimension.
    edge_feat_dim: int
        Size of edge feature dimension.
    hid_dim: int
        Size of hidden dimension.
    out_dim: int
        Size of output dimension.
    num_layers: int
        Number of graph convolutional layers.
    dropout: float
        Dropout rate. Default is 0.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    pooling: str
        Type of ('sum', 'mean', 'max') pooling layer. Default is 'mean'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    lean_beta: bool
        Whether beta is a learnable weight. Default is False.
    aggr: str
        Type of aggregator scheme ('softmax', 'power'). Default is 'softmax'.
    mlp_layers: int
        Number of MLP layers in message normalization. Default is 1.
    """

    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hid_dim,
                 num_layers,
                 dropout=0,
                 norm='layer',
                 beta=1.0,
                 learn_beta=True,
                 aggr='softmax',
                 mlp_layers=1,
                 num_timesteps=2,
                 ):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.node_encoder = nn.Linear(node_in_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_in_dim,hid_dim)

        for i in range(self.num_layers):
            conv = GENConv(in_dim=hid_dim,
                           out_dim=hid_dim,
                           aggregator=aggr,
                           beta=beta,
                           learn_beta=learn_beta,
                           mlp_layers=mlp_layers,
                           norm = norm
                           )

            self.gcns.append(conv)
            self.norms.append(norm_layer(norm, hid_dim))

        self.readout=  AttentiveFPReadout(
            hid_dim, num_timesteps = num_timesteps, dropout = dropout
        )
        self.out = nn.Sequential(
            nn.Linear(hid_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, g):
        node_feats, edge_feats = g.ndata["node_feat"], g.edata["edge_feat"]

        with g.local_scope():
            hv = self.node_encoder(node_feats)
            # hv += self.float_node_embeddings(node_float_feats)
            he = self.edge_encoder(edge_feats)

            for layer in range(self.num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv

            out = self.readout(g, hv)
            out = self.out(out)
            return out


if __name__ == "__main__":
    from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
    def print_param(model):
        print(f"all params: {count_parameters(model)}\n"
              f"trainable params: {count_trainable_parameters(model)}\n"
              f"freeze params: {count_no_trainable_parameters(model)}\n")


    from dataset import SMRTDatasetOneHot
    test_dataset = SMRTDatasetOneHot(name= "SMRT_test_demo", raw_dir="D:\DEEPGNN_RT\dataset")
    test_dataloader = GraphDataLoader(test_dataset, batch_size=6)
    g = test_dataset[0][0]



    # #DeepGCN model test
    # model = DeeperGCN(164, 11, 200, 5, 0.1)
    # print(model)
    # out = model(g)
    # print(out)



    #afp model test
    # model = AttentivfFPModel(164, 11, 200, 200, 5, 200, 0.1)
    # print(model)
    # print_param(model)
    # out = model(g)
    # print(out)



    # # gin model test
    # from dataset import get_node_dim, get_edge_dim
    # full_atom_feature_dims = get_node_dim()
    # full_bond_feature_dims = get_edge_dim()
    #
    # model = GINModel(full_atom_feature_dims, full_bond_feature_dims, num_layers=8)
    # print_param(model)
    # out = model(g)
    # print(out)



    # #gat test
    # model = GATModel(164)
    # print(model)
    # print_param(model)
    # out = model(g)
    # print(out)



    # #gcn test
    # model = GCNModel(164)
    # print(model)
    # print_param(model)
    # out = model(g)
    # print(out)



    # '''ablation model'''
    # model = GCNModelAFPreadout(node_in_dim=164, hidden_feats=[200] * 16)
    # print(model)
    # out = model(g)
    # print(out)
    #
    #
    # model = DeeperGCN(node_in_dim=164, edge_in_dim=11, hid_dim=200, num_layers=16, dropout=0.1, mlp_layers=1)
    # from dgl.nn import SumPooling
    # model.readout = SumPooling()
    # print(model)
    # out = model(g)
    # print(out)
    #
    #
    # model = DeeperGCN(node_in_dim=164, edge_in_dim=11, hid_dim=200, num_layers=16, dropout=0.1, mlp_layers=1)
    # model.out = nn.Linear(200, 1)
    # print(model)
    # out = model(g)
    # print(out)




