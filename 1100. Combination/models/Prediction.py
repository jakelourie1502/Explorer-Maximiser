
import sys
from tracemalloc import start
sys.path.append(".")
import time
import torch
import torch.nn.functional as TF 
from models.res_block import resBlock
class Prediction(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self,device,cfg):
        super().__init__()
        self.cfg = cfg
        self.state_size = self.cfg.state_size
        self.device = device
        self.state_channels = self.cfg.model.state_channels
        self.beta_conv = torch.nn.Conv2d(self.state_channels+1,self.state_channels,1,1,0)
        self.beta_bn = torch.nn.BatchNorm2d(self.state_channels)
        self.resBlocks = torch.nn.ModuleList([resBlock(x,self.device,self.cfg) for x in self.cfg.prediction.res_block])
        
        #value
        self.value_conv = torch.nn.Conv2d(self.state_channels,self.cfg.prediction.value_conv_channels,1,1,0)
        self.bn1v = torch.nn.BatchNorm2d(self.cfg.prediction.value_conv_channels)
        self.FChidV = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.cfg.prediction.value_conv_channels, self.cfg.prediction.value_hidden_dim)
        self.bn2v = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FCoutV = torch.nn.Linear(self.cfg.prediction.value_hidden_dim, self.cfg.prediction.value_support[2])
        
        
        #policy
        self.policy_conv = torch.nn.Conv2d(self.state_channels,self.cfg.prediction.policy_conv_channels,1,1,0)
        self.bn1p = torch.nn.BatchNorm2d(self.cfg.prediction.policy_conv_channels)
        self.FChidp = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.cfg.prediction.policy_conv_channels, self.cfg.prediction.policy_hidden_dim)
        self.bn2p = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FCoutP = torch.nn.Linear(self.cfg.prediction.policy_hidden_dim, self.cfg.actions_size)
        
        #unclaimed novelty
        self.expV_conv = torch.nn.Conv2d(self.state_channels,self.cfg.prediction.expV_conv_channels,1,1,0)
        self.bn1expV = torch.nn.BatchNorm2d(self.cfg.prediction.expV_conv_channels)
        self.FChidexpV = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.cfg.prediction.expV_conv_channels, self.cfg.prediction.expV_hidden_dim)
        self.bn2expV = torch.nn.BatchNorm1d(self.cfg.prediction.expV_hidden_dim)
        self.FCoutexpV = torch.nn.Linear(self.cfg.prediction.expV_hidden_dim, self.cfg.prediction.expV_support[2])
        
        #activation
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.sm_v = torch.nn.Softmax(dim=1)
        self.sm_p = torch.nn.Softmax(dim=1)
        self.sm_n = torch.nn.Softmax(dim=1)

    def forward(self,state,rdn_beta, only_predict_expV=False):
        """RDN beta is a (bn,) shaped vector of betas."""
        rdn_beta_plane = torch.zeros((state.shape[0],1,state.shape[2],state.shape[3])).to(self.device) + torch.tensor(rdn_beta).reshape(state.shape[0],1,1,1) #creates a channel window image, for each element in batch size
        rdn_beta_plane = rdn_beta_plane.float()
        state = torch.cat((state,rdn_beta_plane),dim=1) #appends this to the state
        state = self.beta_conv(state)
        state = self.beta_bn(state)
        state = self.relu(state)
        
        for block in self.resBlocks:
          state = block(state)
        
        if not only_predict_expV:
          ##policy
          p = state
          p = self.policy_conv(p)
          p = self.bn1p(p)
          p = self.relu(p)
          p = torch.flatten(p, start_dim=1)
          p = self.FChidp(p)
          p = self.bn2p(p)
          p = self.relu(p)
          p = self.FCoutP(p)
          p = self.sm_p(p)
          
          ##value
          v = state 
          v = self.value_conv(v)
          v = self.bn1v(v)
          v = self.relu(v)
          v = torch.flatten(v, start_dim=1)
          v = self.FChidV(v)
          v = self.bn2v(v)
          v = self.relu(v)
          v = self.FCoutV(v)
          v = self.sm_v(v)
        else:
          p = v = False
        ##nov
        n = state
        n = self.expV_conv(n)
        n = self.bn1expV(n)
        n = self.relu(n)
        n = torch.flatten(n, start_dim=1)
        n = self.FChidexpV(n)
        n = self.bn2expV(n)
        n = self.relu(n)
        n = self.FCoutexpV(n)
        n = self.sm_n(n)

        return p, v, n