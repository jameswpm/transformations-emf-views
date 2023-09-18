#PyEcore
from pyecore.ecore import EReference, EAttribute

#PyG
from torch_geometric.data import HeteroData

class Model2Graph():

    def __init__(self):        
        self.data = None

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
        # obtain graph
        nodes = {}
        i = 0
         # Initialize a HeteroData object
        self.data = HeteroData()   

        # Create dictionaries to store node indices for different types
        node_indices = {}

        for obj in list_elements:
            if (metafilter is not None) and (not metafilter.pass_filter_object(obj)):
                continue
            # Register node
            node_type = obj.eClass.name
            if not obj in nodes:
                node_indices[node_type] = i
                i = i + 1
                self.data[f'{node_type}'].x = []  # Initialize node features
            # node_index = node_indices[node_type]
            # self.data[f'{node_type}'].x.append(node_index)  # Add node feature

            attributes = {}
            for f in obj.eClass.eAllStructuralFeatures():
                if f.derived:
                    continue
                if (metafilter is not None) and (not metafilter.pass_filter_structural(f)):
                    continue
                # references
                if isinstance(f, EReference):
                    if f.many:
                        for innerObj in obj.eGet(f):
                            if innerObj is None:  # or innerObj.eIsProxy
                                continue
                            # avoid adding elements thar are not in the model
                            if not innerObj in list_elements:
                                continue
                            if ((metafilter is not None) and
                                    (not metafilter.pass_filter_object(innerObj))):
                                continue
                            if not innerObj in nodes:
                                nodes[innerObj] = i
                                i = i + 1
                                G.add_node(nodes[innerObj], type=innerObj.eClass.name)
                            G.add_edge(nodes[obj], nodes[innerObj], type=f.name)
                    else:
                        innerObj = obj.eGet(f)
                        if innerObj is None:  # or innerObj.eIsProxy
                            continue
                        # avoid adding elements thar are not in the model
                        if not innerObj in list_elements:
                            continue
                        if ((metafilter is not None) and
                                (not metafilter.pass_filter_object(innerObj))):
                            continue
                        if not innerObj in nodes:
                            nodes[innerObj] = i
                            i = i + 1
                            G.add_node(nodes[innerObj], type=innerObj.eClass.name)
                        G.add_edge(nodes[obj], nodes[innerObj], type=f.name)
                # attributes
                elif isinstance(f, EAttribute):
                    if f.many:
                        list_att_val = []
                        for innerObj in obj.eGet(f):
                            if innerObj is None:  # or innerObj.eIsProxy
                                list_att_val.append('<none>')
                            else:
                                list_att_val.append(innerObj)
                        attributes[f.name] = list_att_val
                    else:
                        innerObj = obj.eGet(f)
                        if innerObj is None:  # or innerObj.eIsProxy
                            attributes[f.name] = '<none>'
                        else:
                            attributes[f.name] = innerObj
            if consider_attributtes:
                G.nodes[nodes[obj]]['atts'] = attributes
        return G
    
    def _add_node(self, type, content):
        if self.data[type].node_id is None:
            self.data[type].node_id = torch.empty()



    