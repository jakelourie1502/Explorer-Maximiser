import torch
import numpy as np 
from global_settings import state_channels, res_block_kernel_size
class resBlock(torch.nn.Module):
    """
    Input: 
      returns a view with the same dims

    """
    def __init__(self,channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = state_channels,out_channels = channels,kernel_size = res_block_kernel_size,stride = 1,padding = res_block_kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.convIdentity = torch.nn.Conv2d(in_channels = channels, out_channels = state_channels, kernel_size = 1, stride = 1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(state_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.convIdentity(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x

