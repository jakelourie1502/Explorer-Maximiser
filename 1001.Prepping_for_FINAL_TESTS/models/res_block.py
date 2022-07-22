from turtle import down
import torch
import numpy as np 
class resBlock(torch.nn.Module):
    """
    Input: 
      returns a view with the same dims

    """
    def __init__(self,channels,device, cfg,downsample=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
          self.stride = 2
        else:
          self.stride = 1
        self.device = device
        self.cfg = cfg
        self.state_channels = self.cfg.model.state_channels
        self.conv1 = torch.nn.Conv2d(in_channels = channels,out_channels = channels * 2 ,kernel_size = self.cfg.model.res_block_kernel_size,stride = self.stride,padding = self.cfg.model.res_block_kernel_size // 2) #, padding_mode='replicate')
        self.bn1 = torch.nn.BatchNorm2d(channels*2)
        self.convIdentity = torch.nn.Conv2d(in_channels = channels * 2, out_channels = channels, kernel_size = 1, stride = 1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()
        if downsample:
          self.convID2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size= 1, stride = 2, padding=0)
          self.bn3 = torch.nn.BatchNorm2d(channels)
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.convIdentity(x)
        x = self.bn2(x)
        if self.downsample:
          identity = self.convID2(identity)
          identity = self.bn3(identity)

        x += identity
        x = self.relu(x)
        return x

