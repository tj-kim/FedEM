# Import Custom Made Victim
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Attack_Metrics import *
import pandas as pd

# Import Relevant Libraries
from transfer_attacks.Transferer import *

class Boundary_Transferer(): 
    """
    - Load all the datasets but separate them
    - Intermediate values of featues after 2 convolution layers
    """
    
    def __init__(self, models_list, dataloader):
        
        self.models_list = models_list
        self.dataloader = dataloader
        
        self.data_x = self.dataloader.
        
    def select_data_point(self):
        """
        Select a single data point to use as comparison of different boundary types
        """
        