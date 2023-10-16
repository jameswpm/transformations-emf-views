# SageConv layer for Link prediction
from os import sep as DIR_SEP, path as osp
from pathlib import Path

# Starts to create the necessary GNN code to deal with the graph
from torch_geometric.utils import (
    remove_self_loops ,
)

from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import precision_score
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch.utils.data import DataLoader

import numpy as np
from torch import nn
# import networkx as nx
from typing import List , Tuple
from torch_geometric.nn import to_hetero
import torch

import matplotlib.pyplot as plt
import torch_geometric.nn as  gnn

if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("gpu up")
else:  
  dev = "cpu"  
device = torch.device(dev)

DATA_PATH = osp.join(Path(__file__).parent, '..', 'data')

graph = torch.load(DATA_PATH + DIR_SEP + 'graph.pt')
graph = T.ToUndirected()(graph)

transform = RandomLinkSplit(
   is_undirected=False,
   add_negative_train_samples=True,
   disjoint_train_ratio=0.35,
   edge_types=[('SRC-State', 'to', 'TGT-State')],
   rev_edge_types=[('TGT-State', 'rev_to', 'SRC-State')])

train,valid,test = transform(graph)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, aggr="add"):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True, aggr=aggr))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    def forward(self, x_i, x_j):
        out = (x_i*x_j).sum(-1)
        return torch.sigmoid(out)
    
    def reset_parameters(self):
      pass

hidden_dimension = 256
model = SAGE(1, hidden_dimension, hidden_dimension, 7, 0.5).to(device)
predictor = DotProductLinkPredictor().to(device)

def create_train_batch(all_pos_train_edges, perm, edge_index):
    # First, we get our positive edges, reshaping them to the form (2, hidden_dimension)               
    pos_edges = all_pos_train_edges[perm].t().to(device)

    # We then sample the negative edges using PyG functionality
    neg_edges = negative_sampling(edge_index, num_nodes=graph['SRC-State'].num_nodes,
                                  num_neg_samples=perm.shape[0], method='dense').to(device)

    # Our training batch is just the positive edges concatanted with the negative ones
    train_edge = torch.cat([pos_edges, neg_edges], dim=1)  

    # Our labels are all 1 for the positive edges and 0 for the negative ones                          
    pos_label = torch.ones(pos_edges.shape[1], )
    neg_label = torch.zeros(neg_edges.shape[1], )
    train_label = torch.cat([pos_label, neg_label], dim=0).to(device)

    return train_edge, train_label
  
def train(model, predictor, x, adj_t, split_edge, loss_fn, optimizer, batch_size, num_epochs, edge_model=False, spd=None):
  # adj_t isn't used everywhere in PyG yet, so we switch back to edge_index for negative sampling
  row, col, edge_attr = adj_t.t().coo()
  edge_index = torch.stack([row, col], dim=0)

  model.train()
  predictor.train()

  model.reset_parameters()
  predictor.reset_parameters()

  all_pos_train_edges = split_edge['train']['edge']
  for epoch in range(num_epochs):
    epoch_total_loss = 0
    for perm in DataLoader(range(all_pos_train_edges.shape[0]), batch_size,
                           shuffle=True):
      optimizer.zero_grad()

      train_edge, train_label = create_train_batch(all_pos_train_edges, perm, edge_index)

      # Use the GNN to generate node embeddings
      if edge_model:
        h = model(x, edge_index, spd)
      else:
        h = model(x, adj_t)

      # Get predictions for our batch and compute the loss
      preds = predictor(h[train_edge[0]], h[train_edge[1]])
      loss = loss_fn(preds, train_label)

      epoch_total_loss += loss.item()

      # Update our parameters
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
      optimizer.step()
    print(f'Epoch {epoch} has loss {round(epoch_total_loss, 4)}')

def accuracy(pred, label):
  pred_rounded = torch.round(pred)
  accu = torch.eq(pred_rounded, label).sum() / label.shape[0]
  accu = round(accu.item(), 4)
  return accu

@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size, edge_model=False, spd=None):
    model.eval()
    predictor.eval()

    if edge_model:
        # adj_t isn't used everywhere in PyG yet, so we switch back to edge_index 
        row, col, edge_attr = adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        h = model(x, edge_index, spd)
    else:
        h = model(x, adj_t)

    pos_eval_edge = split_edge['edge'].to(device)
    neg_eval_edge = split_edge['edge_neg'].to(device)

    pos_eval_preds = []
    for perm in DataLoader(range(pos_eval_edge.shape[0]), batch_size):
        edge = pos_eval_edge[perm].t()
        pos_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_eval_pred = torch.cat(pos_eval_preds, dim=0)

    neg_eval_preds = []
    for perm in DataLoader(range(neg_eval_edge.size(0)), batch_size):
        edge = neg_eval_edge[perm].t()
        neg_eval_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_eval_pred = torch.cat(neg_eval_preds, dim=0)

    total_preds = torch.cat((pos_eval_pred, neg_eval_pred), dim=0)
    labels = torch.cat((torch.ones_like(pos_eval_pred), torch.zeros_like(neg_eval_pred)), dim=0)
    acc = accuracy(total_preds, labels)

    results = {}
    for K in [10, 20, 30, 40, 50]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_eval_pred,
            'y_pred_neg': neg_eval_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (valid_hits)
    results['Accuracy'] = acc

    return results


optimizer = torch.optim.Adam(
            list(model.parameters())  +
            list(predictor.parameters()), lr=0.01)
train(model, predictor, emb, adj_t, split_edge, torch.nn.BCELoss(), 
      optimizer, 64 * 1024, 30)
test(model, predictor, emb, adj_t, split_edge["valid"], Evaluator(name='ogbl-ddi'), 64*1024)