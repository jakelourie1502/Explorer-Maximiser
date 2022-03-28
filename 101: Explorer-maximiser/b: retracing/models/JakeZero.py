import sys
sys.path.append(".")
import torch
import torch.nn.functional as TF 
import torch.nn as nn
from global_settings import observable_size as env_size, state_channels, proj_l1, proj_out, pred_hid,RND_output_vector

class JakeZero(torch.nn.Module):
    
    def __init__(self, representation, dynamic, prediction):
        super().__init__()
        self.representation_network = representation
        self.dynamic_network = dynamic
        self.prediction_network = prediction
        self.projection = nn.Sequential(nn.Linear(env_size[0]*env_size[1]*state_channels, proj_l1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(proj_l1),
                                    nn.Linear(proj_l1, proj_l1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(proj_l1),
                                    nn.Linear(proj_l1, proj_out),
                                    nn.BatchNorm1d(proj_out))
        self.projection_head1 = nn.Sequential(nn.Linear(proj_out, pred_hid),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(pred_hid),
                                    nn.Linear(pred_hid, proj_out))
        self.RDN = nn.Sequential(nn.Conv2d(state_channels,state_channels,3,1,1),
                                nn.ReLU(),
                                nn.Conv2d(state_channels,state_channels//2,1,1,0),
                                nn.ReLU(),
                                nn.Flatten(start_dim=1),
                                nn.Linear(env_size[0]*env_size[1]*state_channels//2, RND_output_vector),
                                nn.Sigmoid())
        self.RDN_prediction = nn.Sequential(nn.Conv2d(state_channels,state_channels,1,1,0),
                                nn.ReLU(),
                                nn.BatchNorm2d(state_channels),
                                nn.Conv2d(state_channels,state_channels//2,3,1,1),
                                nn.ReLU(),
                                nn.BatchNorm2d(state_channels//2),
                                nn.Flatten(start_dim=1),
                                nn.Linear(env_size[0]*env_size[1]*state_channels//2, RND_output_vector,bias=False),
                                nn.Sigmoid())
                                

    def representation(self, x):
        return self.representation_network(x)

    def dynamic(self, state, action):
        return self.dynamic_network(state, action)
    
    def prediction(self, x):
        return self.prediction_network(x)
    
    def project(self, state, grad_branch=True):
        state = nn.Flatten(start_dim=1)(state)
        proj = self.projection(state)
        if grad_branch:
            return self.projection_head1(proj)
        else:
            return proj.detach()
    
    
