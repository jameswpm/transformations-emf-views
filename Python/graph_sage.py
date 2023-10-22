# SageConv layer for Link prediction
from os import sep as DIR_SEP, path as osp
from pathlib import Path

# Starts to create the necessary GNN code to deal with the graph
from torch_geometric.utils import (
    remove_self_loops,
)

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch.utils.data import DataLoader

import numpy as np
from torch import Tensor, nn
# import networkx as nx
from typing import List, Tuple
from torch_geometric.nn import to_hetero
import torch

import matplotlib.pyplot as plt
import torch_geometric.nn as gnn

if torch.cuda.is_available():
    dev = "cuda:0"
    print("gpu up")
else:
    dev = "cpu"
device = torch.device(dev)

DATA_PATH = osp.join(Path(__file__).parent, '..', 'data')

graph = torch.load(DATA_PATH + DIR_SEP + 'graph.pt')
graph = T.ToUndirected()(graph)

# print(f'Node types:\n {graph.node_types}')
# print(f'Edge types:\n {graph.edge_types}')
# print('Info on the target link')
# print(graph['SRC-State', 'to', 'TGT-State'].num_edges)
# print(graph['TGT-State', 'rev_to', 'SRC-State'].num_edges)

print(graph)

transform = T.RandomLinkSplit(
    is_undirected=False,
    # training (80%), validation (10%), and testing edges (10%).
    num_val=0.1,
    num_test=0.1,
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    disjoint_train_ratio=0.3,
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly.
    neg_sampling_ratio=2.0,
    add_negative_train_samples=True,
    edge_types=[('SRC-State', 'to', 'TGT-State')],
    rev_edge_types=[('TGT-State', 'rev_to', 'SRC-State')])

train, valid, test = transform(graph)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # A 2-layer GNN computation graph.
        # `ReLU` is the non-lineary function used in-between.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:


class Classifier(torch.nn.Module):
    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # # Convert node embeddings to edge-level representations:
        # edge_feat_user = x_user[edge_label_index[0]]
        # edge_feat_movie = x_movie[edge_label_index[1]]

        # # Apply dot-product to get a prediction per supervision edge:
        # return (edge_feat_user * edge_feat_movie).sum(dim=-1)
        out = (x_i * x_j).sum(-1)
        return torch.sigmoid(out)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # initial embeddings
        self.x_i_emb = torch.nn.Embedding(
            graph["SRC-State"].num_nodes, hidden_channels)
        self.x_j_emb = torch.nn.Embedding(
            graph["TGT-State"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:

        self.gnn = to_hetero(self.gnn, metadata=graph.metadata())

        self.classifier = Classifier()

    def forward(self, data) -> Tensor:
        x_dict = {
            "SRC-State": self.x_i_emb(data["SRC-State"].node_id),
            "TGT-State": self.x_j_emb(data["TGT-State"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["SRC-State"],
            x_dict["TGT-State"],
            data['SRC-State', 'to', 'TGT-State'].edge_label_index,
        )

        return pred


model = Model(hidden_channels=64)

print(model)
