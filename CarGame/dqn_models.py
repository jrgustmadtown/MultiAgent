"""
DQN Model architectures
"""

import torch
import copy
from config import LAYER_SIZES


def create_dqn_model():
    """Create a standard DQN model with the architecture defined in config"""
    l1 = LAYER_SIZES['l1']
    l2 = LAYER_SIZES['l2']
    l3 = LAYER_SIZES['l3']
    l4 = LAYER_SIZES['l4']
    
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    return model


def create_target_network(model):
    """Create a copy of the model to use as target network"""
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    return model2


def sync_target_network(source_model, target_model):
    """Synchronize target network with source network parameters"""
    target_model.load_state_dict(source_model.state_dict())
