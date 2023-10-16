from os import sep as DIR_SEP, walk as oswalk, path as osp
from pathlib import Path
import torch

#PyEcore
from pyecore.resources import URI

#Internal
from graph.model2graph import Model2Graph
from modeling.metamodels import Metamodels
from modeling.traces import Traces

RESOURCES_PATH = osp.join(Path(__file__).parent, '..', 'resources')

# Register the metamodels in the resource set
metamodels = Metamodels(osp.join(RESOURCES_PATH, 'metamodels'))
metamodels.register()

# Get the resource set with the registered metamodels
resource_set = metamodels.get_resource_set()

# Set input and output directories
directory_input_src = osp.join(RESOURCES_PATH, 'models', 'yakindu_input')
directory_input_target = osp.join(RESOURCES_PATH, 'models', 'statecharts_output')

model_to_graph = Model2Graph()
    
for subdir, dirs, files in oswalk(directory_input_src):
    for file in files:
      # if file != '100.xmi' and file != '101.xmi':
      #    continue

      filepath = subdir + DIR_SEP + file

      if filepath.endswith(".xmi"):
          xmi_path_src = filepath
          xmi_path_target = directory_input_target + DIR_SEP + file
          xmi_path_traces = directory_input_target + DIR_SEP + "trace_" + file
          # just include in the graph when have all files (src, target and traces)
          if osp.isfile(xmi_path_target) and osp.isfile(xmi_path_traces):
            #For each of main models, save resource with UUIDs at temporary directory
            m_resource_src = resource_set.get_resource(URI(xmi_path_src))
            m_resource_src.use_uuid = True
            m_resource_src.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'src.xmi')))

            model_to_graph.get_graph_from_model(m_resource_src, label="SRC")

            m_resource_target = resource_set.get_resource(URI(xmi_path_target))
            m_resource_target.use_uuid = True
            m_resource_target.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'target.xmi')))

            model_to_graph.get_graph_from_model(m_resource_target, label="TGT")

            # get the traces model to define the target edge_index (edge used for the link prediction)
            m_resource_traces = resource_set.get_resource(URI(xmi_path_traces))

            # Use the traces class to get a mapping of the traces by class
            traces = Traces(m_resource_traces)
            # TODO: State should not be hard-coded
            mapping = traces.get_mapping_traces('State')

            #iterate over traces, adding the edges
            nodes_mapping = model_to_graph.get_nodes()
            for src_uuid, target_uuid in mapping.items():
                src_id = None
                tgt_id = None
                for node_type, node_list in nodes_mapping.items():
                  src = next((item for item in node_list if item["uuid"] == src_uuid), None)
                  tgt = next((item for item in node_list if item["uuid"] == target_uuid), None)
                  if src is not None:
                    src_type = node_type
                    src_id = src['id']
                  if tgt is not None:
                    tgt_type = node_type
                    tgt_id = tgt['id']
                  if src_id is not None and tgt_id is not None:
                     break

                model_to_graph._add_edge(f'{src_type}|to|{tgt_type}', src_id, tgt_id)

merged_graph = model_to_graph.get_hetero_graph()
# merged_graph.generate_ids()
print(merged_graph.validate())

exit()
# Starts to create the necessary GNN code to deal with the graph
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops ,
)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,roc_auc_score, average_precision_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import time


import numpy as np
import scipy.sparse as sp
from torch import nn
import pandas as pd
# import networkx as nx
from typing import List , Dict,Tuple
from torch_geometric.nn import SAGEConv, to_hetero
import torch

import matplotlib.pyplot as plt
import torch_geometric.nn as  gnn
from torch_geometric.data  import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("gpu up")
else:  
  dev = "cpu"  
device = torch.device(dev)

merged_graph = T.ToUndirected()(merged_graph)

transform = RandomLinkSplit(
   is_undirected=False,
   add_negative_train_samples=True,
   disjoint_train_ratio=0.35,
   edge_types=[('state', 'to', 'state')],
   rev_edge_types=[('state', 'rev_to', 'state')])

train,valid,test = transform(merged_graph)

print(train)

#verify edge numbers
print(  
  len(train['state', 'to', 'state'].edge_index[0])+ # train visible message passing edges 
  len(train['state', 'to', 'state'].edge_label) /2  + # train positive supervision edges 
  len(valid['state', 'to', 'state'].edge_label) /2 +  # validation positive evaluation edges
  len(test['state', 'to', 'state'].edge_label) /2  )  # test positive evaluation edges
print(len(merged_graph['state', 'to', 'state'].edge_index[0]) )  # all existing target edges in the graph

print(merged_graph.edge_types)


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
        z = torch.cat([z_dict['state'][edge_label_index[0]], z_dict['state'][edge_label_index[1]]], dim=-1)

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

node_types = [("Y-State",11), ("Y-Transition",67), ("S-State",28)]

myGAE = MyGAE(Encoder(), EdgeDecoder(64), merged_graph, node_types, node_dim=64)
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
    target = train['state', 'to', 'state'].edge_label
    label_edge_index = train['state', 'to', 'state'].edge_label_index
    h = myGAE.encode(train.x_dict,train.edge_index_dict)
    loss  = myGAE.recon_loss(h,label_edge_index[:,target==1],label_edge_index[:,target==0])
    loss.backward()
    optimizer.step()
    auc, p ,acc= myGAE.test(h, train['state', 'to', 'state'].edge_label_index,train['state', 'to', 'state'].edge_label)
    # print(i,"train", np.array([auc, p , acc]).round(3),end=' ')
    
    with torch.no_grad():
      myGAE.eval()
      test.to(device)
      z = myGAE.encode(test.x_dict,test.edge_index_dict)
      auc, p ,acc= myGAE.test(z, test['state', 'to', 'state'].edge_label_index,test['state', 'to', 'state'].edge_label)
      # auc, p = myGAE.test(z, valid.edge_label_index[:,valid.edge_label==1],valid.edge_label_index[:,valid.edge_label==0])
      # print("valid",np.array([auc, p , acc]).round(3))
      validationMetrics.append([auc, p , acc])

plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,0],label='auc')
plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,1],label='precision')
plt.plot(np.arange(len(validationMetrics)),np.array(validationMetrics)[:,2],label='acc')
plt.legend()
plt.show()
