from audioop import bias
from contextlib import suppress
import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
import torch.nn as nn
import numpy as np
from config import Config
from models.Dynamic import Dynamic
from models.Prediction import Prediction
from models.Representation import Representation

from math import sqrt

class ExpMax(torch.nn.Module):
    
    def init_weights_rdn(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0,std=(1/2))
            torch.nn.init.normal_(m.bias, mean=0,std=(0.01/2))

    def __init__(self, device):


        super().__init__()
        self.cfg = Config()
        self.device = device
        self.representation_network = Representation(device).to(device)
        self.dynamic_network = Dynamic(device).to(device)
        self.prediction_network = Prediction(device).to(device)
        
        self.state_size = self.cfg.state_size
        self.env_size = self.cfg.observable_size
        self.state_channels = self.cfg.model.state_channels
        self.proj_l1 = self.cfg.siam.proj_l1
        self.pred_hid = self.cfg.siam.pred_hid
        self.proj_out = self.cfg.siam.proj_out
        self.projection = nn.Sequential(nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.proj_l1),
                                    nn.BatchNorm1d(self.proj_l1),
                                    nn.ReLU(),
                                    nn.Linear(self.proj_l1, self.proj_l1),
                                    nn.BatchNorm1d(self.proj_l1),
                                    nn.ReLU(),
                                    nn.Linear(self.proj_l1, self.proj_out),
                                    nn.BatchNorm1d(self.proj_out))
        self.projection_head1 = nn.Sequential(nn.Linear(self.proj_out, self.pred_hid),
                                    nn.BatchNorm1d(self.pred_hid),
                                    nn.ReLU(),
                                    nn.Linear(self.pred_hid, self.proj_out))
        
        self.RDN = nn.Sequential(nn.Conv2d(self.state_channels,self.state_channels // 2,1,1,0),
                                nn.LeakyReLU(),
                                nn.Conv2d(self.state_channels // 2,self.state_channels,1,1,0),
                                nn.LeakyReLU(),
                                nn.Flatten(start_dim=1),
                                nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.RND_output_vector),
                                
        )
        
        self.RDN.apply(self.init_weights_rdn)    
        
        self.RDN_prediction = nn.Sequential(nn.Conv2d(self.state_channels,self.state_channels // 2,1,1,0),
                                nn.LeakyReLU(),
                                nn.Conv2d(self.state_channels // 2,self.state_channels,1,1,0),
                                nn.LeakyReLU(),
                                nn.Flatten(start_dim=1),
                                nn.Linear(self.state_size[0]*self.state_size[1]*self.state_channels, self.cfg.RND_output_vector*2),
                                nn.LeakyReLU(),
                                nn.Linear(self.cfg.RND_output_vector*2, self.cfg.RND_output_vector),
        )
        
    
    def representation(self, x):
        return self.representation_network(x)

    def dynamic(self, state, action):
        return self.dynamic_network(state, action)
    
    def prediction(self, x,rdn_beta):
        return self.prediction_network(x,rdn_beta)
    
    
        
    def project(self, state, grad_branch=True):
        state = nn.Flatten(start_dim=1)(state)
        proj = self.projection(state)
        if grad_branch:
            return self.projection_head1(proj)
        else:
            return proj.detach()
    