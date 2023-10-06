from os import sep as DIR_SEP, walk as oswalk, path as osp
from pathlib import Path
import torch

#PyEcore
from pyecore.resources import URI
from modeling.metamodels import Metamodels

#Internal
from graph.model2graph import Model2Graph
from pyecore.resources import ResourceSet, URI
from modeling.traces import Traces

RESOURCES_PATH = osp.join(Path(__file__).parent, '..', 'resources', 'models')

def get_graph_from_models(xmi_path_src, xmi_path_target, xmi_path_traces):
  #For each of main models, get the graph representation
  m_resource_src = resource_set.get_resource(URI(xmi_path_src))
  m_resource_src.use_uuid = True
  # #save resource with UUIDs at temporary directory
  m_resource_src.save(output=URI(osp.join(RESOURCES_PATH, 'temp', 'src.xmi')))

  m_resource_target = resource_set.get_resource(URI(xmi_path_target))
  m_resource_target.use_uuid = True
  m_resource_target.save(output=URI(osp.join(RESOURCES_PATH, 'temp', 'target.xmi')))

#   model_to_graph_src = Model2Graph(label="SRC")
#   model_to_graph_src.get_graph_from_model(m_resource_src)

#   model_to_graph_target = Model2Graph(label="TGT")
#   model_to_graph_target.get_graph_from_model(m_resource_target)

#   # merge the two graphs
#   merged_graph = model_to_graph_src.get_hetero_graph().update(model_to_graph_target.get_hetero_graph())

  # get the traces model to define the target edge_index (edge used for the link prediction)
  m_resource_traces = resource_set.get_resource(URI(xmi_path_traces))
  # m_resource_traces.use_uuid = True
  # temp_path = osp.join(RESOURCES_PATH, 'temp', 'traces.xmi')
  # uri_traces = URI(temp_path)
  # m_resource_traces.save(output=uri_traces)

  # Use the traces class to get a mapping of the traces by class
  traces = Traces(m_resource_traces)
  # TODO: State should not be hard-coded
  mapping = traces.get_mapping_traces('State')

  print(mapping)

# Register the metamodels in the resource set
metamodels = Metamodels(osp.join(RESOURCES_PATH, '..','metamodels'))
metamodels.register()

# Get the resource set with the registered metamodels
resource_set = metamodels.get_resource_set()

xmi_path_src = osp.join(RESOURCES_PATH, 'yakindu_input', '100.xmi')
#xmi_path_target = osp.join(RESOURCES_PATH, 'persons.xmi')
xmi_path_target = osp.join(RESOURCES_PATH, 'statecharts_output', '100.xmi')
# xmi_path_trace = osp.join(RESOURCES_PATH, 'traces.xmi')
xmi_path_trace = osp.join(RESOURCES_PATH, 'statecharts_output', 'trace_100.xmi')

get_graph_from_models(xmi_path_src, xmi_path_target, xmi_path_trace)