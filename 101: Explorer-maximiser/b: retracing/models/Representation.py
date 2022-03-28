
import torch
import numpy as np 
from global_settings import hidden_layer_rep_channels, state_channels, hidden_layer_rep_params, to_state_params,res_block_reps
from models.res_block import resBlock
class Representation(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self):
        self.hidden_layer_channels = hidden_layer_rep_channels
        self.state_channels = state_channels
        self.hidden_layer_params = hidden_layer_rep_params
        self.to_state_params = to_state_params
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 2,out_channels = self.hidden_layer_channels,
                                    kernel_size = self.hidden_layer_params['kernel_size'],
                                    stride = self.hidden_layer_params['stride'],
                                    padding = self.hidden_layer_params['padding'])
        self.bn1 = torch.nn.BatchNorm2d(self.hidden_layer_channels)
        self.conv2 = torch.nn.Conv2d(in_channels = self.hidden_layer_channels,out_channels = self.state_channels,
                                    kernel_size = self.to_state_params['kernel_size'],
                                    stride = self.to_state_params['stride'],
                                    padding = self.to_state_params['padding'])
        self.bn2 = torch.nn.BatchNorm2d(self.state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x) for x in res_block_reps])
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        for block in self.resBlocks:
          x = block(x)
        return x