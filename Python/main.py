from os import path as osp
from pathlib import Path
import torch

#PyEcore
from pyecore.resources import URI
from graph.model2graph import Model2Graph

#Internal
from modeling.metamodels import Metamodels
from modeling.traces import Traces

RESOURCES_PATH = osp.join(Path(__file__).parent, '..', 'resources')

 # Register the metamodels in the resource set
metamodels = Metamodels(osp.join(RESOURCES_PATH, 'metamodels'))
metamodels.register()

resource_set = metamodels.get_resource_set()

#For each of main models, get the graph representation
xmi_path_left = osp.join(RESOURCES_PATH, 'models', 'yakindu.xmi')
m_resource_left = resource_set.get_resource(URI(xmi_path_left))
m_resource_left.use_uuid = True
# #save resource with UUIDs at temporary directory
m_resource_left.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'yakindu.xmi')))

xmi_path_right = osp.join(RESOURCES_PATH, 'models', 'statecharts-emftvm.xmi')
m_resource_right = resource_set.get_resource(URI(xmi_path_right))
m_resource_right.use_uuid = True
m_resource_right.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'statecharts-emftvm.xmi')))

#TODO: How to define the labels?
model_to_graph_left = Model2Graph(label="Y")
model_to_graph_left.get_graph_from_model(m_resource_left)

model_to_graph_right = Model2Graph(label="S")
model_to_graph_right.get_graph_from_model(m_resource_right)

# merge the two graphs to be able to create trace links
merged_graph = model_to_graph_left.get_hetero_graph().update(model_to_graph_right.get_hetero_graph())

# get the graph of the trace model
model_to_graph_traces = Model2Graph()
xmi_path_traces = osp.join(RESOURCES_PATH, 'models', 'yakindu_stratecharts_traces.xmi')
m_resource_traces = resource_set.get_resource(URI(xmi_path_traces))
m_resource_traces.use_uuid = True
m_resource_traces.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'yakindu_stratecharts_traces.xmi')))

# Use the traces class to get a mapping of the traces by class
traces = Traces(m_resource_traces)
mapping = traces.get_mapping_traces('State')

#iterate over traces, adding the edge in the merged graph
edges = []
left_nodes_mapping = model_to_graph_left.get_mapping_nodes()
right_nodes_mapping = model_to_graph_right.get_mapping_nodes()
for src_uuid, target_uuid in mapping.items():
    edges.append([left_nodes_mapping[src_uuid], right_nodes_mapping[target_uuid]])
merged_graph['state', 'to', 'state'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print(merged_graph.edge_types)

# Starts to create the necessary GNN code to deal with the graph
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops ,
)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.transforms import RandomLinkSplit
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





