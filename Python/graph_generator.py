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

i = 0
for subdir, dirs, files in oswalk(directory_input_src):
    for file in files:

      if i >= 10:
         break
      
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
      i += 1

merged_graph = model_to_graph.get_hetero_graph()

if merged_graph.validate():
   # Save the generated graph in a external file
   data_dir = osp.join(Path(__file__).parent, '..', 'data')
   torch.save(merged_graph, data_dir + DIR_SEP + 'graph.pt')