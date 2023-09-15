from os import path as osp
from pathlib import Path

from modeling.metamodels import Metamodels


METAMODELS_PATH = osp.join(Path(__file__).parent, '..', 'resources')

 # Register the metamodels in the resource set
metamodels = Metamodels()
metamodels.register()



