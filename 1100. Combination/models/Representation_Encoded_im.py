import torch
import numpy as np 
from config import Config
from models.res_block import resBlock
class Representation(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self, device,cfg):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.state_channels = self.cfg.model.state_channels

        self.conv1 = torch.nn.Conv2d(in_channels = self.cfg.deque_length,out_channels = self.cfg.repr.conv1['channels'],
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0)
        self.bn1 = torch.nn.BatchNorm2d(self.cfg.repr.conv1['channels'])
        self.conv2 = torch.nn.Conv2d(in_channels = self.cfg.repr.conv1['channels'],out_channels = self.state_channels,
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0)
        self.bn2 = torch.nn.BatchNorm2d(self.state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x,self.device,self.cfg, False) for x in self.cfg.repr.res_block_channels[:1]])
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