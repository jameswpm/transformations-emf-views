from os import path as osp
from pathlib import Path

#PyEcore
from pyecore.resources import URI
from graph.model2graph import Model2Graph

#Internal
from modeling.metamodels import Metamodels
from graph.encoder import Enconder

RESOURCES_PATH = osp.join(Path(__file__).parent, '..', 'resources')

 # Register the metamodels in the resource set
metamodels = Metamodels(osp.join(RESOURCES_PATH, 'metamodels'))
metamodels.register()

resource_set = metamodels.get_resource_set()

graph_encoder = Enconder(embeddings_information=None)
# Register the models in the resource set
xmi_path_left = osp.join(RESOURCES_PATH, 'models', 'yakindu.xmi')
m_resource_left = resource_set.get_resource(URI(xmi_path_left))
m_resource_left.use_uuid = True
# #save resource with UUIDs at temporary directory
# m_resource_left.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'yakindu.xmi')))

xmi_path_right = osp.join(RESOURCES_PATH, 'models', 'statecharts-emftvm.xmi')
m_resource_right = resource_set.get_resource(URI(xmi_path_right))
# m_resource_right.use_uuid = True
# m_resource_right.save(output=URI(osp.join(RESOURCES_PATH, 'models', 'temp', 'statecharts-emftvm.xmi')))

model_to_graph = Model2Graph()
model_to_graph.get_graph_from_model(m_resource_left)
print(model_to_graph.get_hetero_graph())





