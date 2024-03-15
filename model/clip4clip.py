import os
from enum import Enum

import torch
from torch import nn

class TemporalMode(Enum):
    MEAN_POOLING = 0
    TRANSFORMER = 1
    # CrossTransformer = 2        # not used.

def build_model(state_dict: dict):
    """ build model from a given state dict
    if state_dict is {} => build from empty
    """
    
class CLIP4Clip(nn.Module):
    ...