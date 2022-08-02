from audioop import bias
from contextlib import suppress
import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
import torch.nn as nn
import numpy as np
from models.Dynamic import Dynamic
from models.Prediction import Prediction
from models.TwoHeadPrediction import Prediction as Prediction2Heads
from models.Representation import Representation
from models.Representation_Encoded_im import Representation as RepEncoded
from models.res_block import resBlock
from math import sqrt

class ExpMax(torch.nn.Module):
    
    def init_weights_rdn(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0,std=(1/2))
            torch.nn.init.normal_(m.bias, mean=0,std=(0.01/2))

    def __init__(self, device,cfg):


        super().__init__()
        self.cfg = cfg
        self.device = device
        if self.cfg.rgb_im:
            self.representation_network = Representation(device,cfg).to(device)
        else:
            self.representation_network = RepEncoded(device, cfg).to(device)
        self.dynamic_network = Dynamic(device,cfg).to(device)
        if self.cfg.use_two_heads:
            self.prediction_network = Prediction2Heads(device,cfg).to(device)
            
        else:
            self.prediction_network = Prediction(device,cfg).to(device)
                    


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
        self.close_state_projection = nn.Sequential(
            nn.Conv2d(self.state_channels, self.state_channels, 1,1,0),
            nn.BatchNorm2d(self.state_channels, momentum=0.1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.cfg.state_size[0]*self.cfg.state_size[1] * self.state_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,128),
        )
        self.close_state_projection_obs = nn.Sequential(
            nn.Conv2d(3, 32,3,2,1), #24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), #12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2,2), #8
            nn.Flatten(),
            nn.Linear(6*6*64,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.close_state_classifer = nn.Sequential(
            nn.Linear(256, 180),
            nn.BatchNorm1d(180),
            nn.ReLU(),
            nn.Linear(180,self.cfg.actions_size),
            nn.Softmax(1),
            
        )
        
        self.state_vecs_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,128)
        )
    def representation(self, x):
        return self.representation_network(x)

    def dynamic(self, state, action):
        return self.dynamic_network(state, action)
    
    def prediction(self, x,rdn_beta, only_predict_expV = False):
        return self.prediction_network(x,rdn_beta, only_predict_expV)
        
    def project(self, state, grad_branch=True):
        state = nn.Flatten(start_dim=1)(state)
        proj = self.projection(state)
        if grad_branch:
            return self.projection_head1(proj)
        else:
            return proj.detach()
    
    def contrast_two_state_vecs(self, x, y):
        x = self.state_vecs_proj(x)
        y = self.state_vecs_proj(y)
        # return torch.nn.CosineSimilarity(dim=1)(x,y).reshape(-1,1) / 0.07
        return torch.nn.CosineSimilarity(dim=1)(x,y).reshape(-1,1)