import torch
import numpy as np 
from models.res_block import resBlock
from utils import normalize_state_vec
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
        self.obs_size = (self.cfg.channels) * (self.cfg.timesteps_in_obs)
        if self.cfg.running_reward_in_obs:
          self.obs_size += self.cfg.timesteps_in_obs - 1
        if self.cfg.store_prev_actions:
          self.obs_size += self.cfg.timesteps_in_obs - 1
        self.conv1 = torch.nn.Conv2d(in_channels = self.obs_size,out_channels = self.cfg.repr.conv1['channels'],
                                    kernel_size = self.cfg.repr.conv1['kernel_size'],
                                    stride = self.cfg.repr.conv1['stride'],
                                    padding = self.cfg.repr.conv1['padding'])
        self.bn1 = torch.nn.BatchNorm2d(self.cfg.repr.conv1['channels'])
        self.conv2 = torch.nn.Conv2d(in_channels = self.cfg.repr.conv1['channels'],out_channels = self.cfg.repr.conv2['channels'],
                                    kernel_size = self.cfg.repr.conv2['kernel_size'],
                                    stride = self.cfg.repr.conv2['stride'],
                                    padding = self.cfg.repr.conv2['padding'])
        self.bn2 = torch.nn.BatchNorm2d(self.cfg.repr.conv2['channels'])
        self.resBlocks = torch.nn.ModuleList([resBlock(x,self.device, self.cfg, y) for x, y  in zip(self.cfg.repr.res_block_channels, self.cfg.repr.res_block_ds)])
        self.avgpool1 = torch.nn.AvgPool2d(2,2)
        self.avgpool2 = torch.nn.AvgPool2d(2,2)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resBlocks[0](x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.resBlocks[1](x)
        x = self.avgpool1(x)
        if self.cfg.atari_env:
          x = self.resBlocks[2](x)
          x = self.avgpool2(x)
        # if self.cfg.norm_state_vecs:
        #   x = normalize_state_vec(x)
        return x