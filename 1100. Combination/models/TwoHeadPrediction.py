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
        self.device = device
        self.state_channels = self.cfg.model.state_channels
        self.state_size = self.cfg.state_size
        
        self.resBlocks = torch.nn.ModuleList([resBlock(x,self.device,self.cfg) for x in self.cfg.prediction.res_block])
        
        self.initial_conv = torch.nn.Conv2d(self.state_channels,self.state_channels,1,1,0)
        self.bn_init = torch.nn.BatchNorm2d(self.state_channels)
        self.beta_conv = torch.nn.Conv2d(self.state_channels+1,self.state_channels,1,1,0)
        self.bn_beta = torch.nn.BatchNorm2d(self.state_channels)
        
        #value
        self.FChidV_expl = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.prediction.value_hidden_dim)
        self.bn2v_expl = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FChidV_maxi = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.prediction.value_hidden_dim)
        self.bn2v_maxi = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FCoutV_maxi = torch.nn.Linear(self.cfg.prediction.value_hidden_dim, self.cfg.prediction.value_support[2])
        self.FCoutV_expl = torch.nn.Linear(self.cfg.prediction.value_hidden_dim, self.cfg.prediction.value_support[2])
        
        #policy
        self.FChidp_expl = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.prediction.policy_hidden_dim)
        self.bn2p_expl = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FChidp_maxi = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.prediction.policy_hidden_dim)
        self.bn2p_maxi = torch.nn.BatchNorm1d(self.cfg.prediction.value_hidden_dim)
        self.FCoutP_maxi = torch.nn.Linear(self.cfg.prediction.policy_hidden_dim, self.cfg.actions_size)
        self.FCoutP_explr = torch.nn.Linear(self.cfg.prediction.policy_hidden_dim, self.cfg.actions_size)

        #unclaimed novelty
        self.FChidexpV = torch.nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.prediction.expV_hidden_dim)
        self.bn2expV = torch.nn.BatchNorm1d(self.cfg.prediction.expV_hidden_dim)
        self.FCoutexpV = torch.nn.Linear(self.cfg.prediction.expV_hidden_dim, self.cfg.prediction.expV_support[2])
        
        #activation
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()
        self.sm_v = torch.nn.Softmax(dim=1)
        self.sm_p = torch.nn.Softmax(dim=1)
        self.sm_n = torch.nn.Softmax(dim=1)

    def forward(self,state,rdn_beta, only_predict_expV = False):
        
        ### Initial global stuff

        state = self.initial_conv(state)
        state = self.relu(state)
        state = self.bn_init(state)
        for block in self.resBlocks:
          state = block(state)
        
        
        flat_state = torch.flatten(state, start_dim=1)
        #### convs for rdn
        rdn_beta_plane = torch.zeros((state.shape[0],1,state.shape[2],state.shape[3])).to(self.device) + torch.tensor(rdn_beta).reshape(state.shape[0],1,1,1) #creates a channel window image, for each element in batch size
        rdn_beta_plane = rdn_beta_plane.float()
        expl_state = torch.cat((state,rdn_beta_plane),dim=1) #appends this to the state
        expl_state = self.beta_conv(expl_state)
        expl_state = self.relu(expl_state)
        expl_state = self.bn_beta(expl_state)
        expl_state = torch.flatten(expl_state, start_dim=1)
        
        if not only_predict_expV:
          ##policy
          p_expl = self.FChidp_expl(expl_state)
          p_expl = self.bn2p_expl(p_expl)
          p_expl = self.relu(p_expl)
          p_expl = self.FCoutP_explr(p_expl)
          p_expl = self.sm_p(p_expl)

          p_maxi = self.FChidp_maxi(flat_state)
          p_maxi = self.bn2p_maxi(p_maxi)
          p_maxi = self.relu(p_maxi)
          p_maxi = self.FCoutP_maxi(p_maxi)
          p_maxi = self.sm_p(p_maxi)
          
          ##value
          v_expl = self.FChidV_expl(expl_state)
          v_expl = self.bn2v_expl(v_expl)
          v_expl = self.relu(v_expl)
          v_expl = self.FCoutV_expl(v_expl)
          v_expl = self.sm_v(v_expl)
          
          
          v_maxi = self.FChidV_maxi(flat_state)
          v_maxi = self.bn2v_maxi(v_maxi)
          v_maxi = self.relu(v_maxi)
          v_maxi = self.FCoutV_maxi(v_maxi)
          v_maxi = self.sm_p(v_maxi)
        
        else:
          p_expl = p_maxi = v_maxi = v_expl = False
        
        ##nov
        n = expl_state
        n = self.FChidexpV(n)
        n = self.relu(n)
        n = self.bn2expV(n)
        n = self.FCoutexpV(n)
        n = self.sm_n(n)

        return p_maxi, p_expl, v_maxi, v_expl, n

