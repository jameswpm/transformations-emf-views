from pyecore.resources import ResourceSet, URI
import glob
from os import listdir, path as osp
from pathlib import Path

class Metamodels():
    
    def __init__(self, path):
        """_summary_
        """
        self.resource_set = ResourceSet()
        self.metamodels_resources_path = glob.glob(path)[0]

    def register(self):
        """_summary_
        """
        files = listdir(self.metamodels_resources_path)
        ecore_paths = [f for f in files if osp.isfile(osp.join(self.metamodels_resources_path, f))]
        
        for ecore_file in ecore_paths:

            ecore_path = osp.join(self.metamodels_resources_path, ecore_file)
            resource_path = self.resource_set.get_resource(URI(ecore_path))
            for content in resource_path.contents:
                                
                if hasattr(content, 'nsURI') and content.nsURI != "":
                    self.resource_set.metamodel_registry[content.nsURI] = content
                
                contents = content.eContents

                for content in contents:
                    if hasattr(content, 'nsURI') and content.nsURI != "":
                        self.resource_set.metamodel_registry[content.nsURI] = content

    def get_resource_set(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self.resource_set