from os import path as osp
from pathlib import Path

#PyEcore
from pyecore.resources import URI

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

xmi_path_right = osp.join(RESOURCES_PATH, 'models', 'statecharts-emftvm.xmi')
m_resource_right = resource_set.get_resource(URI(xmi_path_right))

data, left_original_mapping, right_original_mapping = graph_encoder.xmi_to_graph(m_resource_left, m_resource_right)





