

class Traces():

    def __init__(self, model_resource):
        """_summary_
        """

        self.model_resource = model_resource
        self.mapping = {}
    
    def get_mapping_traces(self, src_class_name):
        for root in self.model_resource.contents:
            for element in root.eAllContents():
                class_name = element.eClass.name
                if class_name == 'TraceLink':
                    #TODO: It works only for 1to1 relation.                   
                    source_element = element.sourceElements[0]
                    object_element = source_element.object
                    object_element_class = object_element.eClass.name
                    if object_element_class == src_class_name:
                        #get the target of the relation
                        target_element = element.targetElements[0].object
                        self.mapping[object_element._internal_id] = target_element._internal_id
        
        return self.mapping

                
