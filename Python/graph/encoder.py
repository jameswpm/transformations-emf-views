from typing import Tuple, Optional
import torch
#import pandas as pd


from torch_geometric.data import HeteroData

#from utils.encoders import IdentityEncoder, SequenceEncoder, EnumEncoder, NoneEncoder

class ToGraph():

    def __init__(self, embeddings_information = None):
        
        # self.features_for_embedding_left = [feature for feature in embeddings_information if feature.startswith("s")]
        # self.features_for_embedding_right = [feature for feature in embeddings_information if feature.startswith("t")]
        self.embeddings_information = embeddings_information


    def xmi_to_graph(self, model_root_left, model_root_right, relations_csv_path, class_left, class_right, relation_name):
        return