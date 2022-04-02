import sys
from tracemalloc import start
sys.path.append(".")
import time
import torch
import torch.nn.functional as TF 
from global_settings import exp_v_support, observable_size as env_size, state_channels,value_conv_channels,policy_conv_channels,res_block_pred, policy_output_supp, policy_hidden_dim, value_hidden_dim, value_support
from models.res_block import resBlock
class Prediction(torch.nn.Module):
    """
    Input: 
      Takes in an observation, and returns a state.
    Notes:
     
    Outputs: 
      a state representation

    """
    def __init__(self):
        super().__init__()
        self.resBlocks = torch.nn.ModuleList([resBlock(x) for x in res_block_pred])
        #value
        self.value_conv = torch.nn.Conv2d(state_channels,value_conv_channels,1,1,0)
        self.bn1v = torch.nn.BatchNorm2d(value_conv_channels)
        self.FChidV = torch.nn.Linear(env_size[0]*env_size[1]*value_conv_channels, value_hidden_dim)
        self.bn2v = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutV_robin = torch.nn.Linear(value_hidden_dim, value_support[2])
        self.FCoutV_batman = torch.nn.Linear(value_hidden_dim, value_support[2])
        
        #policy
        self.policy_conv = torch.nn.Conv2d(state_channels,policy_conv_channels,1,1,0)
        self.bn1p = torch.nn.BatchNorm2d(policy_conv_channels)
        self.FChidp = torch.nn.Linear(env_size[0]*env_size[1]*policy_conv_channels, policy_hidden_dim)
        self.bn2p = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutP_robin = torch.nn.Linear(policy_hidden_dim, policy_output_supp)
        self.FCoutP_batman = torch.nn.Linear(policy_hidden_dim, policy_output_supp)
        
        #unclaimed novelty
        self.nov_conv = torch.nn.Conv2d(state_channels,value_conv_channels,1,1,0)
        self.bn1nov = torch.nn.BatchNorm2d(value_conv_channels)
        self.FChidnov = torch.nn.Linear(env_size[0]*env_size[1]*value_conv_channels, value_hidden_dim)
        self.bn2nov = torch.nn.BatchNorm1d(value_hidden_dim)
        self.FCoutnov = torch.nn.Linear(value_hidden_dim, exp_v_support[2])
        

        #activation
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.sm_v = torch.nn.Softmax(dim=1)
        self.sm_p = torch.nn.Softmax(dim=1)
        self.sm_n = torch.nn.Softmax(dim=1)

    def forward(self,state):
        # x = torch.flatten(state,start_dim=1)
        for block in self.resBlocks:
          state = block(state)
        
        ##policy
        p = state
        p = self.policy_conv(p)
        
        
        p = self.bn1p(p)
        p = self.relu(p)
        p = torch.flatten(p, start_dim=1)
        p = self.FChidp(p)
        p = self.bn2p(p)
        p = self.relu(p)
        p_robin = self.FCoutP_robin(p)
        p_robin = self.sm_p(p_robin)
        p_batman = self.FCoutP_batman(p)
        p_batman = self.sm_p(p_batman)

        ##value
        v = state 
        v = self.value_conv(v)
        v = self.bn1v(v)
        v = self.relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.FChidV(v)
        v = self.bn2v(v)
        v = self.relu(v)
        v_robin = self.FCoutV_robin(v)
        v_robin = self.sm_v(v_robin)
        v_batman = self.FCoutV_batman(v)
        v_batman = self.sm_v(v_batman)

        ##nov
        n = state
        n = self.nov_conv(n)
        n = self.bn1nov(n)
        n = self.relu(n)
        n = torch.flatten(n, start_dim=1)
        n = self.FChidnov(n)
        n = self.bn2nov(n)
        n = self.relu(n)
        n = self.FCoutnov(n)
        n = self.sm_n(n)
        return p_robin, p_batman,v_robin, v_batman, n