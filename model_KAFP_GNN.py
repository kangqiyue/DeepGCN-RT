import os
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

full_atom_feature_dims = 164
full_bond_feature_dims = 11


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(full_atom_feature_dims, self.emb_dim)

    def forward(self, x):
        x_embedding = self.linear(x)
        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.linear_bond = nn.Linear(full_bond_feature_dims, self.emb_dim)

    def forward(self, x):
        bond_embedding = self.linear_bond(x)
        return bond_embedding


# AttentiveFP
# Code was adapted from DGL-LifeSci, url: https://lifesci.dgl.ai/_modules/dgllife/model/gnn/attentivefp.html#AttentiveFPGNN

class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.relu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GNN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        # node_feats = self.layer_norm(node_feats)
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.relu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a)
    :param B: has shape (b, k)
    :return: (b, aK)
    """
    assert A.dim() == 2 and B.dim() == 2
    res = torch.einsum('ba,bk->bak', A, B).view(A.size(0),
                                                A.size(1) * B.size(1)
                                                )
    return res


class KroneckerMessage(nn.Module):
    def __init__(self, graph_feat_size, out_dim, dropout):
        super(KroneckerMessage, self).__init__()
        #KroneckerMessage
        self.proj_node= nn.Sequential(
            nn.Linear(graph_feat_size, 20),
            nn.LayerNorm(20),
            nn.ReLU(),
        )

        self.proj_kron = nn.Sequential(
            nn.Linear(400, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, g, node_feat):
        g.ndata["node_proj"] = self.proj_node(node_feat)
        g.apply_edges(self.apply_edges_kron)
        g.edata["kron"] = self.proj_kron(g.edata["kron"])

        g.update_all(message_func=fn.copy_edge("kron", "kron"), reduce_func=fn.sum('kron', 'kron_feat'))
        out = g.ndata["kron_feat"]
        return out

    def apply_edges_kron(self, edges):
        return {'kron': kronecker_product_einsum_batched(edges.src['node_proj'], edges.dst['node_proj'])}


class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.

    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.ReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.ReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.ReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.ReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.layer_norm = nn.LayerNorm(graph_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])
        return self.layer_norm(self.attentive_gru(g, logits, node_feats))


class GNNLayerKAFP(nn.Module):
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, out_dim, dropout):
        super(GNNLayerKAFP, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.ReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.layer_norm = nn.LayerNorm(graph_feat_size)


        self.kron = KroneckerMessage(graph_feat_size, out_dim, dropout)

        self.concat_proj = nn.Sequential(
            nn.Linear(graph_feat_size + out_dim, graph_feat_size),
            nn.LayerNorm(graph_feat_size),
            nn.ReLU()
        )


    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        node_feats_kron = self.kron(g, node_feats)

        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])
        node_feats_gru = self.attentive_gru(g, logits, node_feats)
        node_feats_gru = self.layer_norm(node_feats_gru)

        output = torch.cat([node_feats_gru, node_feats_kron], dim=1)
        output = self.concat_proj(output)
        return output

from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
class AttentiveFPGNNLN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 num_layers=5,
                 graph_feat_size=200,
                 dropout=0.1):
        super(AttentiveFPGNNLN, self).__init__()
        self.init_context = GetContext(node_in_dim, edge_in_dim, graph_feat_size, dropout)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

        # readout
        self.readout = AttentiveFPReadout(graph_feat_size, dropout=dropout)
        # linear layer
        self.out = nn.Sequential(
            nn.Linear(graph_feat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        # node_feats =self.atom_encoder(torch.tensor(g.ndata["node_feat"], dtype = torch.int64).to(device))
        # edge_feats = self.bond_encoder(torch.tensor(g.edata["edge_feat"], dtype = torch.int64).to(device))
        # node_feats = self.atom_encoder(g.ndata["node_feat"])
        # edge_feats = self.bond_encoder(g.edata["edge_feat"])
        node_feats = g.ndata["node_feat"]
        edge_feats = g.edata["edge_feat"]

        node_feats = self.init_context(g, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        feats = self.readout(g, node_feats)

        return self.out(feats)


class KAFPGNN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 num_layers=5,
                 graph_feat_size=200,
                 out_dim=40,
                 dropout=0.1):
        super(KAFPGNN, self).__init__()
        self.init_context = GetContext(node_in_dim, edge_in_dim, graph_feat_size, dropout)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayerKAFP(graph_feat_size, graph_feat_size, out_dim, dropout))

        # readout
        self.readout = AttentiveFPReadout(graph_feat_size, dropout=dropout)
        # linear layer
        self.out = nn.Sequential(
            nn.Linear(graph_feat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = g.ndata["node_feat"]
        edge_feats = g.edata["edge_feat"]

        node_feats = self.init_context(g, node_feats, edge_feats)

        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        feats = self.readout(g, node_feats)

        return self.out(feats)


if __name__ == '__main__':
    from train_func import count_parameters, count_no_trainable_parameters, count_trainable_parameters
    def print_param(model):
        print(f"all params: {count_parameters(model)}\n"
              f"trainable params: {count_trainable_parameters(model)}\n"
              f"freeze params: {count_no_trainable_parameters(model)}\n")


    from dataset import SMRTDatasetOneHot
    test_dataset = SMRTDatasetOneHot(name= "SMRT_test_demo", raw_dir="D:\Molecule\DEEPGNN_RT\dataset")
    test_dataloader = GraphDataLoader(test_dataset, batch_size=6)
    g = test_dataset[0][0]

    model = AttentiveFPGNNLN(164, 11 ,graph_feat_size=200, dropout=0.1, num_layers=3)
    print(model)
    print_param(model)
    model(g)
    model = KAFPGNN(164, 11 ,graph_feat_size=200,out_dim=40, dropout=0.1, num_layers=3)
    print(model)
    print_param(model)
    out = model(g)
    print(out)
