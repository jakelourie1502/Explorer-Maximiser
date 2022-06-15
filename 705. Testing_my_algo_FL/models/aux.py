from xxlimited import Xxo
import torch
import numpy as np 
from config import Config
from models.res_block import resBlock
class AuxValueOutput(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self, device):
        super().__init__()
        self.cfg = Config()
        self.device = device
        self.state_channels = self.cfg.model.state_channels

        self.conv1 = torch.nn.Conv2d(in_channels = self.cfg.deque_length,out_channels = self.cfg.aux.conv1['channels'],
                                    kernel_size = self.cfg.aux.conv1['kernel_size'],
                                    stride = self.cfg.aux.conv1['stride'],
                                    padding = self.cfg.aux.conv1['padding'])
        self.bn1 = torch.nn.BatchNorm2d(self.cfg.aux.conv1['channels'])
        self.conv2 = torch.nn.Conv2d(in_channels = self.cfg.aux.conv1['channels'],out_channels = self.cfg.aux.conv2['channels'],
                                    kernel_size = self.cfg.aux.conv2['kernel_size'],
                                    stride = self.cfg.aux.conv2['stride'],
                                    padding = self.cfg.aux.conv2['padding'])
        self.bn2 = torch.nn.BatchNorm2d(self.cfg.aux.conv2['channels'])
        self.value_conv = torch.nn.Conv2d(self.cfg.aux.conv2['channels']+1, self.cfg.aux.conv2['channels'], 3, 1, 1)
        self.bn3 = torch.nn.BatchNorm2d(self.cfg.aux.conv2['channels'])
        self.value_lin1 = torch.nn.Linear(self.cfg.aux.conv2['channels']*self.cfg.observable_size[0]*self.cfg.observable_size[1], self.cfg.aux.value_hidden_dim)
        self.value_lin2 = torch.nn.Linear(self.cfg.aux.value_hidden_dim, self.cfg.prediction.value_support[2])
        
        
        self.reward_conv = torch.nn.Conv2d(self.cfg.aux.conv2['channels']+1, self.cfg.aux.conv2['channels'], 3, 1, 1)
        self.bn4 = torch.nn.BatchNorm2d(self.cfg.aux.conv2['channels'])
        self.reward_lin1 = torch.nn.Linear(self.cfg.aux.conv2['channels']*self.cfg.observable_size[0]*self.cfg.observable_size[1], self.cfg.aux.reward_hidden_dim)
        self.reward_lin2 = torch.nn.Linear(self.cfg.aux.reward_hidden_dim, self.cfg.dynamic.reward_support[2])

        self.sm = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()

    def forward(self,x,action, rdn_beta):
        x=x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        #value
        v = x
        rdn_beta_plane = torch.zeros((v.shape[0],1,v.shape[2],v.shape[3])).to(self.device) + torch.tensor(rdn_beta).reshape(v.shape[0],1,1,1) #creates a channel window image, for each element in batch size
        rdn_beta_plane = rdn_beta_plane.float()
        v = torch.cat((v,rdn_beta_plane),dim=1) #appends this to the state
        v = self.value_conv(v)
        v = self.relu(v)
        v = self.bn3(v)
        v = torch.nn.Flatten(start_dim=1)(v)
        v = self.value_lin1(v)
        v = self.relu(v)
        v = self.value_lin2(v)
        v = self.sm(v)

        #reward
        r = x
        action = torch.tensor(action+1) / self.cfg.actions_size
        action = action.reshape(-1,1,1,1)
        action_plane = torch.zeros(r.shape[0],1, r.shape[2], r.shape[3]).to(self.device).float()
        action_plane += action
        r = torch.cat((r,action_plane),dim=1)
        r = self.reward_conv(r)
        r = self.relu(r)
        r = self.bn4(r)
        r = torch.nn.Flatten(start_dim=1)(r)
        r = self.reward_lin1(r)
        r = self.relu(r)
        r = self.value_lin2(r)
        r = self.sm(r)

        return r, v 