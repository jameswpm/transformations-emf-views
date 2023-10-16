from os import sep as DIR_SEP, path as osp
from pathlib import Path

# Starts to create the necessary GNN code to deal with the graph
from torch_geometric.utils import (
    remove_self_loops ,
)

from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import precision_score
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

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

#verify edge numbers
print(  
  len(train['SRC-State', 'to', 'TGT-State'].edge_index[0])+ # train visible message passing edges 
  len(train['SRC-State', 'to', 'TGT-State'].edge_label) /2  + # train positive supervision edges 
  len(valid['SRC-State', 'to', 'TGT-State'].edge_label) /2 +  # validation positive evaluation edges
  len(test['SRC-State', 'to', 'TGT-State'].edge_label) /2  )  # test positive evaluation edges
print(len(graph['SRC-State', 'to', 'TGT-State'].edge_index[0]) )  # all existing target edges in the graph

class Encoder(nn.Module):
  """
  basic gnn with two layers to get representations for every node  
  """
  def __init__(self, node_dim=64):
    super(Encoder,self).__init__()   
    self.gconv1  = gnn.SAGEConv((-1,-1), node_dim)
    self.gconv2  = gnn.SAGEConv((-1,-1), node_dim)

  def forward(self, x, edge_index):
    h = self.gconv1(x, edge_index).relu()
    h = self.gconv2(h, edge_index)
    return h

EPS = 1e-15

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        z = torch.cat([z_dict['SRC-State'][edge_label_index[0]], z_dict['TGT-State'][edge_label_index[1]]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).sigmoid()
        return z.view(-1)


class MyGAE(nn.Module):

  def __init__(self, encoder,decoder, data:HeteroData, node_types:List[Tuple], node_dim:int):

    super(MyGAE,self).__init__()
    self.node_types = {}
    for n in node_types:
      self.node_types[n[0]]  = nn.Embedding(n[1], node_dim) # this part is redundant(we can use embeddings from torch geometric directly)
      self.node_types[n[0]].to(device)
    self.hetero_encoder = to_hetero(encoder, data.metadata())
    self.decoder = decoder

  def encode(self, nodes, edges):
    xVectors= {}
    for i in self.node_types.keys():
      xVectors[i] = self.node_types[i](nodes[i])
    return self.hetero_encoder(xVectors, edges)
    
  def decode(self,h,edge_label_index):
    
    return self.decoder(h,edge_label_index)

  def recon_loss(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        """

        pos_loss = -torch.log(
            self.decode(z, pos_edge_index) + EPS).mean()

        # Do not include self-loops in negative samples
        # pos_edge_index, _ = remove_self_loops(pos_edge_index)
        # pos_edge_index, _ = add_self_loops(pos_edge_index)
        neg_edge_index, _ = remove_self_loops(neg_edge_index)
        
        neg_loss = -torch.log(1 -
                              self.decode(z, neg_edge_index) +
                              EPS).mean()

        return pos_loss + neg_loss

  def test(self,z, edge_index, y ):
    r"""Given latent variables :obj:`z`, edges
    computes area under the ROC curve (AUC) , average precision (AP)
    and (Acc) accuracy scores  .
    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        edge_index (LongTensor):   edges
        y  (Tensor):labels
    """
    pred =  self.decode(z, edge_index,)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), precision_score(y, pred>=0.5), accuracy_score(y,pred>0.5)

node_types = [("SRC-State", 2473), ("TGT-State", 5343)]

myGAE = MyGAE(Encoder(), EdgeDecoder(64), graph, node_types, node_dim=64)
myGAE.to(device)
train = train.to(device)
with torch.no_grad():
    myGAE.encode(train.x_dict, train.edge_index_dict)

optimizer = torch.optim.Adam(myGAE.parameters(), lr=0.01, weight_decay=0.0005)

validationMetrics = []
for i in range(40):
    myGAE.train()
    optimizer.zero_grad()
    # pred = myGAE(train.x_dict, train.edge_index_dict,
    #              train['Gene', "associates",'Disease'].edge_label_index)
    target = train['SRC-State', 'to', 'TGT-State'].edge_label
    label_edge_index = train['SRC-State', 'to', 'TGT-State'].edge_label_index
    h = myGAE.encode(train.x_dict,train.edge_index_dict)
    loss  = myGAE.recon_loss(h,label_edge_index[:,target==1],label_edge_index[:,target==0])
    loss.backward()
    optimizer.step()
    auc, p ,acc= myGAE.test(h, train['SRC-State', 'to', 'TGT-State'].edge_label_index,train['SRC-State', 'to', 'TGT-State'].edge_label)
    # print(i,"train", np.array([auc, p , acc]).round(3),end=' ')
    
    with torch.no_grad():
      myGAE.eval()
      test.to(device)
      z = myGAE.encode(test.x_dict,test.edge_index_dict)
      auc, p ,acc= myGAE.test(z, test['SRC-State', 'to', 'TGT-State'].edge_label_index,test['SRC-State', 'to', 'TGT-State'].edge_label)
      # auc, p = myGAE.test(z, valid.edge_label_index[:,valid.edge_label==1],valid.edge_label_index[:,valid.edge_label==0])
      # print("valid",np.array([auc, p , acc]).round(3))
      validationMetrics.append([auc, p , acc])

plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,0],label='auc')
plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,1],label='precision')
plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,2],label='acc')
plt.legend()
plt.show()
