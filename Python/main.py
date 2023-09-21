from os import path as osp
from pathlib import Path

#PyEcore
from pyecore.resources import URI
from graph.model2graph import Model2Graph

#Internal
from modeling.metamodels import Metamodels

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
# m_resource_left.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'yakindu.xmi')))

xmi_path_right = osp.join(RESOURCES_PATH, 'models', 'statecharts-emftvm.xmi')
m_resource_right = resource_set.get_resource(URI(xmi_path_right))
# m_resource_right.use_uuid = True
# m_resource_right.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'statecharts-emftvm.xmi')))

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
model_to_graph_traces.get_graph_from_model(m_resource_traces)

print(model_to_graph_traces.get_hetero_graph())





