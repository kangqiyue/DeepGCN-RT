#data transformation
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, List
import torch
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs,load_graphs

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

import feature_ops as feature_ops
from loguru import logger
import traceback



def get_edge_dim(exclude_feature=None):
    """Hacky way to get edge dim from bond_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    if exclude_feature:
        edge_dim = len(feature_ops.bond_featurizer(mol.GetBonds()[0], exclude_feature))
    else:
        edge_dim = len(feature_ops.bond_featurizer(mol.GetBonds()[0], exclude_feature))

    return edge_dim

def get_node_dim(exclude_feature=None):
    """Hacky way to get node dim from atom_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    if exclude_feature:
        node_dim = len(feature_ops.atom_featurizer(mol.GetAtoms()[0], exclude_feature))
    else:
        node_dim = len(feature_ops.atom_featurizer(mol.GetAtoms()[0], exclude_feature))
    return node_dim

def get_node_features(mol, exclude_feature=None):
    node_features = np.array([
        feature_ops.atom_featurizer(atom,exclude_feature) for atom in mol.GetAtoms()
    ], dtype='float32')
    # logger.info({"node_dim": node_features, "exclude_feature": exclude_feature})
    return node_features

def get_edge_features(mol, exclude_feature=None):
    edge_features = np.array([
        feature_ops.bond_featurizer(bond, exclude_feature) for bond in mol.GetBonds()
    ], dtype="float32"
    )
    # logger.info({"node_dim": edge_features, "exclude_feature": exclude_feature})
    return edge_features


"""adopted and modified from ogb;
url: https://github.com/snap-stanford/ogb/blob/c8f0d2aca80a4f885bfd6ad5258ecf1c2d0ac2d9/ogb/utils/mol.py#L6"""
def smiles2graph(smiles_string,  exclude_node, exclude_edge):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    x = get_node_features(mol, exclude_node)
    # bond
    num_bond_features = get_edge_dim(exclude_edge)  #edge_dim
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = feature_ops.bond_featurizer(bond, exclude_edge)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int32).T
        edge_attr = np.array(edge_features_list, dtype= np.float32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.float32)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def feature_to_dgl_graph(graph):
    '''
    input
    -------
    graph_dict, example:
    {'edge_feat': array([[0, 0, 0],
        [0, 0, 0]], dtype=int64), 'edge_index': array([[0, 1],
        [1, 0]], dtype=int64), 'node_feat': array([[5, 0, 4, 5, 3, 0, 2, 0, 0],
        [7, 0, 2, 5, 1, 0, 2, 0, 0]], dtype=int64), 'num_nodes': 2}
    output
    ---------
    #dgl_graph(int 32)
    '''
    g = dgl.graph((graph["edge_index"][0, :], graph["edge_index"][1, :]), num_nodes=graph["num_nodes"], idtype=torch.int32)
    g.ndata['node_feat'] = torch.tensor(graph['node_feat'], dtype=torch.float32)
    g.edata["edge_feat"] = torch.tensor(graph['edge_feat'], dtype=torch.float32)

    return g


def smile_to_dgl(x_smiles):
    graph_list = []
    for i in range(len(x_smiles)):
        g = feature_to_dgl_graph(smiles2graph(x_smiles[i]))
        #     nx_plot(g)
        graph_list.append(g)
        if i % 1000 == 0:
            print(i)


class SMRTDatasetOneHot(DGLDataset):
    # def __init__(self, raw_dir="/data/users/kangqiyue/kqy/DEEPGNN_RT/dataset", force_reload=False, verbose=False, name='SMRT_test', ):
    #     super(SMRTDatasetOneHot, self).__init__(name=name,
    #                                       raw_dir=raw_dir,
    #                                       force_reload=force_reload,
    #                                       verbose=verbose)

    def __init__(self, name, url=None, raw_dir=None, save_dir=None,
                 hash_key=(), force_reload=False, verbose=False, transform=None, exclude_node=None, exclude_edge=None):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()
        self._transform = transform

        # if no dir is provided, the default dgl download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()
        else:
            self._raw_dir = raw_dir

        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir
        self.exclude_node = exclude_node
        self.exclude_edge = exclude_edge

        self._load()


    def process(self):
        filename = self.name + "_set.txt"
        self.graphs, self.label = self._load_graph(filename)

    def _load_graph(self, filename):
        self.train_set = pd.read_csv(os.path.join(self.raw_dir, filename), sep="\t")
        # self.test_set = pd.read_csv(os.path.join(self.raw_dir, "SMRT_test_set.txt"), sep="\t")

        data = self.train_set
        x_smiles = data["smiles"]
        g_labels = data["RT"]
        g_labels = torch.tensor(g_labels, dtype=torch.float32)
        graph_list = []
        for i in range(len(x_smiles)):
            g = feature_to_dgl_graph(smiles2graph(x_smiles[i], exclude_node=self.exclude_node, exclude_edge=self.exclude_edge))
            # nx_plot(g)
            graph_list.append(g)
            if i % 1000 == 0:
                print(i)
        return graph_list, g_labels

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, 'dgl_graph_SMRT.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})
        print(f"data saved in: {graph_path}")

    def has_cache(self):
        # graph_path = os.path.join(self.save_path, 'dgl_graph_SMRT.bin')
        # return os.path.exists(graph_path)
        return False

    def load(self):
        print(f"loading data from: {os.path.join(self.save_path, 'dgl_graph_SMRT.bin')}")
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph_SMRT.bin'))
        self.graphs = graphs
        self.label = label_dict['labels']

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)


def load_smrt_data_one_hot(random_state=42, demo = False, raw_dir = "/data/users/kangqiyue/kqy/DEEPGNN_RT/dataset", exclude_node=None, exclude_edge=None):
    if demo:
        train_dataset = SMRTDatasetOneHot(name="SMRT_train_demo", raw_dir=raw_dir, exclude_node=exclude_node, exclude_edge=exclude_edge)
        train_dataset, valid_dataset = dgl.data.utils.split_dataset(train_dataset, [0.9, 0.1], shuffle=True,random_state=random_state)
        test_dataset = SMRTDatasetOneHot(name="SMRT_test_demo", raw_dir=raw_dir, exclude_node=exclude_node, exclude_edge=exclude_edge)
    else:
        train_dataset = SMRTDatasetOneHot(name="SMRT_train", raw_dir = raw_dir, exclude_node=exclude_node, exclude_edge=exclude_edge)
        train_dataset, valid_dataset = dgl.data.utils.split_dataset(train_dataset, [0.9, 0.1], shuffle=True, random_state=random_state)
        test_dataset = SMRTDatasetOneHot(name="SMRT_test", raw_dir = raw_dir, exclude_node=exclude_node, exclude_edge=exclude_edge)

    print(f"----len----\n"
          f"demo is {demo}\n"
          f"len of train_dataset: {len(train_dataset)}\n"
          f"len of valid_dataset: {len(valid_dataset)}\n"
          f"len of test_dataset: {len(test_dataset)}\n"
          )
    return train_dataset, valid_dataset, test_dataset


class TLDataset(SMRTDatasetOneHot):
    def __init__(self, name, raw_dir="D:\yue\chem_dataset\DEEPGNN_RT\dataset\\10_subdataset\processed"):
        super(TLDataset, self).__init__(name=name, raw_dir=raw_dir)

    def process(self):
        filename = self.name + ".xlsx"
        self.graphs, self.label = self._load_graph(filename)

    def _load_graph(self, filename):
        self.dataset = pd.read_excel(os.path.join(self.raw_dir, filename), engine='openpyxl')
        data = self.dataset
        x_smiles = data["smiles"]
        g_labels = data["rt"]
        g_labels = torch.tensor(g_labels, dtype=torch.float32)
        graph_list = []
        for i in tqdm(range(len(x_smiles))):
            g = feature_to_dgl_graph(smiles2graph(x_smiles[i]))
            # nx_plot(g)
            graph_list.append(g)
            if i % 1000 == 0:
                print(i)
        return graph_list, g_labels

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})
        print(f"data saved in: {graph_path}")

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def load(self):
        print(f"loading data from: {os.path.join(self.save_path, 'dgl_graph.bin')}")
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
        self.graphs = graphs
        self.label = label_dict['labels']


class RikenDataset(TLDataset):
    def __init__(self, name , raw_dir):
        super().__init__(name= name, raw_dir=raw_dir)

    def process(self):
        filename = self.name + ".csv"
        self.graphs, self.label = self._load_graph(filename)

    def _load_graph(self, filename):
        self.dataset = pd.read_csv(os.path.join(self.raw_dir, filename))
        data = self.dataset
        x_smiles = data["pred_smiles"]
        g_labels = data["rt"]
        g_labels = torch.tensor(g_labels, dtype=torch.float32)
        graph_list = []
        for i in tqdm(range(len(x_smiles))):
            g = feature_to_dgl_graph(smiles2graph(x_smiles[i]))
            # nx_plot(g)
            graph_list.append(g)
            if i % 1000 == 0:
                print(i)
        return graph_list, g_labels

if __name__ == "__main__":
    test = TLDataset(name= "Riken_Training", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN")
    test = TLDataset(name= "Riken_Test", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN")
    test = TLDataset(name= "RIKEN_External", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN")
    test = TLDataset(name= "Training_HILIC", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN_HILIC")
    test = TLDataset(name= "Test_HILIC", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN_HILIC")
    test = TLDataset(name= "Test_HILIC", raw_dir="D:\DEEPGNN_RT\dataset\RIKEN_HILIC")


    tl_list = [  'Cao_HILIC_116',
                 'Eawag_XBridgeC18_364',
                 'FEM_lipids_72',
                 'FEM_long_412',
                 'FEM_short_73',
                 'IPB_Halle_82',
                 'LIFE_new_184',
                 'LIFE_old_194',
                 'MTBLS87_147',
                 'UniToyama_Atlantis_143']
    for t in tl_list:
        test_tl = TLDataset(name=t)
        print(len(test_tl))

    print("transfer to graph finished!")









