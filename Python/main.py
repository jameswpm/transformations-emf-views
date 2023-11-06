from os import sep as DIR_SEP, path as osp
from pathlib import Path

# Starts to create the necessary GNN code to deal with the graph

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_score
import torch_geometric.transforms as T
import torch.nn.functional as F

import numpy as np
from torch.nn import Embedding
from torch_geometric.nn import to_hetero, SAGEConv
from torch import Tensor
import torch

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
    print("gpu up")
else:
    dev = "cpu"
device = torch.device(dev)

DATA_PATH = osp.join(Path(__file__).parent, '..', 'data')

graph = torch.load(DATA_PATH + DIR_SEP + 'graph.pt')
graph = T.ToUndirected()(graph)

if not graph.validate():
    raise "The graph is not valid"

transform = T.RandomLinkSplit(
    is_undirected=False,
    # using default PyG split
    num_val=0.1,
    num_test=0.2,
    # Across the training edges, we use 65% of edges for message passing,
    # and 35% of edges for supervision.
    disjoint_train_ratio=0.35,
    add_negative_train_samples=True,
    edge_types=[('SRC_State', 'to', 'TGT_State')],
    rev_edge_types=[('TGT_State', 'rev_to', 'SRC_State')]
)

train_data, valid_data, test_data = transform(graph)

# check the split of the target edge in the generated graphs
# train_data visible message passing edges + train_data positive supervision edges + validation positive evaluation edges+test positive evaluation edges
sum_edges = (len(train_data['SRC_State', 'to', 'TGT_State'].edge_index[0]) +
             len(train_data['SRC_State', 'to', 'TGT_State'].edge_label) / 2 +
             len(valid_data['SRC_State', 'to', 'TGT_State'].edge_label) / 2 +
             len(test_data['SRC_State', 'to', 'TGT_State'].edge_label) / 2)
# all existing target edges in the graph
all_edges = len(graph['SRC_State', 'to', 'TGT_State'].edge_index[0])

if float(sum_edges) != float(all_edges):
     raise f'The graph split is not valid. Sum of split: {sum_edges} and All edges: {all_edges}'

print(graph)

"""
Definition of the GNN with the SageConv Layers and the Classifier using dot-product
"""
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1,-1), hidden_channels)
        self.conv2 = SAGEConv((-1,-1), hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # A 2-layer GNN computation graph.
        # `ReLU` is the non-lineary function used in-between.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_i: Tensor, x_j: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_a = x_i[edge_label_index[0]]
        edge_feat_b = x_j[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_a * edge_feat_b ).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # initial embeddings
        self.node_types = {}
        for n in graph.node_types:
            self.node_types[n]  = Embedding(graph[n].num_nodes, hidden_channels)
            self.node_types[n].to(device)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=graph.metadata())

        self.classifier = Classifier()

    def forward(self, data) -> Tensor:
        x_dict = {}
        for i in self.node_types.keys():
            input_tensor = data[i].node_id
            if not torch.all(input_tensor >= 0) or not torch.all(input_tensor < self.node_types[i].num_embeddings):
                # Using torch.clamp() to clamp the indices to the valid range when the node tensor is wrong defined.
                input_tensor = torch.clamp(input_tensor, 0, self.node_types[i].num_embeddings - 1)
            x_dict[i] = self.node_types[i](input_tensor)

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["SRC_State"],
            x_dict["TGT_State"],
            data['SRC_State', 'to', 'TGT_State'].edge_label_index,
        )

        return pred

model = Model(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_losses=[]
total_loss = 0
total_examples = 0

def train():
    global train_losses
    global total_loss
    global total_examples

    model.train()
    optimizer.zero_grad()

    # Run `forward` pass of the model
    pred = model.forward(train_data)
    # Apply binary cross entropy
    loss = F.binary_cross_entropy_with_logits(pred, train_data['SRC_State', 'to', 'TGT_State'].edge_label)

    train_losses += [loss]

    loss.backward()
    optimizer.step()
    total_loss += float(loss) * pred.numel()
    total_examples += pred.numel()
    return float(loss)

preds = []
ground_truths = []

@torch.no_grad()
def test(data):
    model.eval()

    # Run `forward` pass of the model to get the prediction
    pred_itr = model.forward(data)

    ground_truth_itr = data['SRC_State', 'to', 'TGT_State'].edge_label
    preds.append(pred_itr)
    ground_truths.append(ground_truth_itr)
    auc = roc_auc_score(ground_truth_itr, pred_itr)
    return float(auc)


for epoch in range(1, 6):
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(valid_data)
    test_rmse = test(test_data)
    # Add checks to handle potential None values
    if loss is not None and train_rmse is not None and val_rmse is not None and test_rmse is not None:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    else:
        print(f'Epoch: {epoch:03d}, Some values are None.')
