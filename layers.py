import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from ogb.graphproppred.mol_encoder import BondEncoder
from dgl.nn.functional import edge_softmax
from modules import MLP, MessageNorm
from dgl.nn.pytorch import GraphConv


class GENConv(nn.Module):
    r"""
    
    Description
    -----------
    Generalized Message Aggregator was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    dataset: str
        Name of ogb dataset.
    in_dim: int
        Size of input dimension.
    out_dim: int
        Size of output dimension.
    aggregator: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 learn_beta=True,
                 p=1.0,
                 learn_p=False,
                 msg_norm=False,
                 learn_msg_scale=False,
                 norm='batch',
                 mlp_layers=2,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for i in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels, norm=norm)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

        # self.edge_encoder = BondEncoder(in_dim)

    def forward(self, g, node_feats, edge_feats, return_edge=False):
        with g.local_scope():
            # Node and edge feature dimension need to match.
            g.ndata['h'] = node_feats
            g.edata['h'] = edge_feats
            g.apply_edges(fn.u_add_e('h', 'h', 'm'))

            if self.aggr == 'softmax':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            
            elif self.aggr == 'power':
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata['m'], minv, maxv)
                g.update_all(lambda edge: {'x': torch.pow(edge.data['m'], self.p)},
                             fn.mean('x', 'm'))
                torch.clamp_(g.ndata['m'], minv, maxv)
                g.ndata['m'] = torch.pow(g.ndata['m'], self.p)
            
            else:
                raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')
            
            if self.msg_norm is not None:
                g.ndata['m'] = self.msg_norm(node_feats, g.ndata['m'])
            
            feats = node_feats + g.ndata['m']
            
            # return self.mlp(feats)
            # return (self.mlp(feats), g.edata["m"]) if return_edge else self.mlp(feats)
            if return_edge:
                return self.mlp(feats), g.edata["m"]
            else:
                return self.mlp(feats)


"adopted and modified from: https://lifesci.dgl.ai/_modules/dgllife/model/gnn/gcn.html#GCN"
class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    output_norm : output normalization
        "layer_norm", "batch_norm", "none"
        default to be "none".
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, out_feats, gnn_norm='both', activation=None,
                 residual=True, output_norm="none", dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=gnn_norm, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        # if residual:
        #     self.res_connection = nn.Linear(in_feats, out_feats)

        if output_norm == "batch_norm":
            self.bn_layer = nn.BatchNorm1d(out_feats)
            self.output_norm = True
        elif output_norm == "layer_norm":
            self.bn_layer = nn.LayerNorm(out_feats)
            self.output_norm = True
        elif output_norm == "none":
            self.output_norm = False
        else:
            raise NotImplementedError

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.output_norm:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        new_feats = self.activation(new_feats)
        new_feats = self.dropout(new_feats)
        if self.residual:
            new_feats = new_feats + feats

        if self.output_norm:
            new_feats = self.bn_layer(new_feats)

        return new_feats


"adopted and modified from: https://lifesci.dgl.ai/_modules/dgllife/model/gnn/gcn.html#GCN"
class GCNLayerWithEdge(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None,
                 residual=True, output_norm="none", dropout=0., update_func="no_relu"):
        super(GCNLayerWithEdge, self).__init__()

        self.activation = activation
        self.mlp = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.aggr = update_func # relu, relu_eps_beta, no_relu

        if self.aggr == "relu_eps_beta":
            #for relu eps beta
            self.eps=1e-7
            self.beta = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        if output_norm == "batch_norm":
            self.bn_layer = nn.BatchNorm1d(out_feats)
            self.output_norm = True
        elif output_norm == "layer_norm":
            self.bn_layer = nn.LayerNorm(out_feats)
            self.output_norm = True
        elif output_norm == "none":
            self.output_norm = False
        else:
            raise NotImplementedError

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature dimension need to match.
            g.ndata['h'] = node_feats
            g.edata['h'] = edge_feats
            g.apply_edges(fn.u_add_e('h', 'h', 'm'))


            if self.aggr == 'relu_eps_beta':
                g.edata['m'] = F.relu(g.edata['m']) + self.eps
                g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            elif self.aggr == "no_relu":
                g.edata['a'] = edge_softmax(g, g.edata['m'])
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            elif self.aggr == "relu":
                # relu activation; have softmax aggration
                g.edata['m'] = F.relu(g.edata['m'])
                g.edata['a'] = edge_softmax(g, g.edata['m'])
                g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                             fn.sum('x', 'm'))
            else:
                raise NotImplementedError

            new_feats = g.ndata['m']
            new_feats = self.mlp(new_feats)
            new_feats = self.activation(new_feats)
            new_feats = self.dropout(new_feats)

            if self.residual:
                new_feats = new_feats + node_feats
            if self.output_norm:
                new_feats = self.bn_layer(new_feats)

            return new_feats