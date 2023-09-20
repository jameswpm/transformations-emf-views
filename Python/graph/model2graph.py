#PyEcore
from pyecore.ecore import EReference, EAttribute

#PyG
import torch
from torch_geometric.data import HeteroData

class Model2Graph():

    def __init__(self):        
        # Initialize a HeteroData object
        self.data = HeteroData()
        # Create list of dictionaries to store nodes, attributes and edges for different types
        self.nodes = {}
        self.nodes_attrs = {}
        self.edge_index = {}

    def get_graph_from_model(self, model_resource, metafilter=None,
                            consider_attributtes=False):
        # traverse the model
        list_elements = []
        for root in model_resource.contents:
            list_elements.append(root)
            list_elements = list_elements + list(root.eAllContents())
        return self._get_graph_from_model_elements(list_elements, metafilter=metafilter,
                                            consider_attributtes=consider_attributtes)


    def _get_graph_from_model_elements(self, list_elements, metafilter=None,
                                    consider_attributtes=False):   

        for obj in list_elements:
            if (metafilter is not None) and (not metafilter.pass_filter_object(obj)):
                continue
            # Add node
            node_type = obj.eClass.name
            node_index = self._add_node(node_type, obj)       

            attributes = {}
            for f in obj.eClass.eAllStructuralFeatures():
                if f.derived:
                    continue
                if (metafilter is not None) and (not metafilter.pass_filter_structural(f)):
                    continue
                # references
                if isinstance(f, EReference):
                    if f.many:
                        for ref_obj in obj.eGet(f):
                            if ref_obj is None:
                                continue
                            # avoid adding elements that are not in the model
                            if not ref_obj in list_elements:
                                continue
                            if ((metafilter is not None) and
                                    (not metafilter.pass_filter_object(ref_obj))):
                                continue
                            ref_obj_type = ref_obj.eClass.name
                            ref_node_index = self._add_node(ref_obj_type, ref_obj)

                            # Add edge
                            self._add_edge(f'{node_type}_{f.name}_{ref_obj_type}', node_index, ref_node_index)
                            
                    else:
                        ref_obj = obj.eGet(f)
                        if ref_obj is None:
                            continue
                        # avoid adding elements thar are not in the model
                        if not ref_obj in list_elements:
                            continue
                        if ((metafilter is not None) and
                                (not metafilter.pass_filter_object(ref_obj))):
                            continue
                        ref_obj_type = ref_obj.eClass.name
                        ref_node_index = self._add_node(ref_obj_type, ref_obj)

                        # Add edge
                        self._add_edge(f'{node_type}_{f.name}_{ref_obj_type}', node_index, ref_node_index)
                # attributes
                elif isinstance(f, EAttribute): #TODO: It lacks some test with models including attributes
                    if f.many:
                        list_attr_val = []
                        for ref_attr in obj.eGet(f):
                            if ref_attr is None:
                                list_attr_val.append('<none>')
                            else:
                                list_attr_val.append(ref_attr)
                        attributes[f.name] = list_attr_val
                    else:
                        ref_attr = obj.eGet(f)
                        if ref_attr is None:
                            attributes[f.name] = '<none>'
                        else:
                            attributes[f.name] = ref_attr
            if consider_attributtes:
                self._add_node_attributtes(node_type, node_index, attributes)
        
        # # Convert edge lists and attributes to PyTorch tensors
        # for edge_type in self.data.edge_types:
        #     self.data[edge_type].edge_index = torch.tensor(self.data[edge_type].edge_index, dtype=torch.long).t().contiguous()
    
    def get_hetero_graph(self):
        return self.data

    def _add_node(self, node_type, node):
        if node_type not in self.nodes:
            self.nodes[node_type] = []
        if node not in self.nodes[node_type]:  
            self.nodes[node_type].append(node)
        #zero index and avoid empty lists
        node_index = len(self.nodes[node_type]) - 1

        return node_index
    
    def _add_node_attributtes(self, node_type, node_index, attr_list):
        if node_type not in self.nodes_attrs:
            self.nodes_attrs[node_type] = {}

        # do not replace a existent list
        if node_index not in self.nodes_attrs[node_type]:
            self.nodes_attrs[node_type][node_index] = attr_list
    
    def _add_edge(self, edge_indentifier, src_index, target_index):
        if edge_indentifier not in self.edge_index:
            self.edge_index[edge_indentifier] = []
        self.edge_index[edge_indentifier].append([src_index, target_index])
    