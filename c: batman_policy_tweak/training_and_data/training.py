from numpy.core.defchararray import upper
import torch
import numpy as np 
from global_settings import exp_v_support, value_support, reward_support, value_only
from utils import scalar_to_support_batch
from collections import deque

def loss_func_r(r, true_values, dones_k, weights):
    """
    r is an M by value_support[2]
    true_values is a M by 1
    weigths is a M by 1
    dones_k is an M by 1
    """
    
    true_values = scalar_to_support_batch(true_values, *reward_support) #M by value_support[2]
    loss = -torch.sum(true_values * torch.log(r+1e-4),dim=1,keepdim=True)
    loss = weights * loss * (1-dones_k)
    return torch.mean(loss)

def loss_func_v(v, true_values, dones_k,weights):
    true_values = scalar_to_support_batch(true_values,*value_support)
    losses = -torch.sum(true_values * torch.log(v+1e-4),dim=1,keepdim=True)
    losses = weights*losses
    if not value_only:
        losses = losses * (1-dones_k)
    return torch.mean(losses)

def loss_func_p(p, true_policy, dones_k,weights):
    losses = torch.sum(true_policy * torch.log2(p + 1e-5),dim=1,keepdim=True)
    losses = -losses * weights * (1-dones_k)
    return torch.mean(losses)

def loss_func_entropy(p):
    return torch.mean(torch.sum(p * (torch.log2(p)+1e-3),dim=1))

def loss_func_proj(stopped_proj, w_grad_head, dones, weights):
    #L1 loss
    contrastive_loss = torch.sum(stopped_proj*w_grad_head,dim=1,keepdims=True)/(
            (torch.sum(stopped_proj**2,dim=1,keepdims=True)*torch.sum(w_grad_head**2,dim=1,keepdims=True))**0.5)
    return -torch.mean((1-dones)*weights*contrastive_loss)

def loss_func_future_nov(nov, true_unclaimed_nov, dones_k, weights):
    true_unclaimed_nov = scalar_to_support_batch(true_unclaimed_nov, *exp_v_support)
    losses = -torch.sum(true_unclaimed_nov * torch.log(nov+1e-4),dim=1,keepdim=True)
    losses = (1-dones_k)*weights*losses
    return torch.mean(losses)


class RDN_loss:
    def __init__(self, mu=1, siggy = 1):
        self.kickoff = True
        self.mu = mu
        self.siggy = siggy
        self.mu_sq = mu**2
        self.logs = 0
        self.a = 0.0001
        self.deki = deque([], 10000)
        self.update_log = []

    def evaluate(self,random_output, predicted_output,log=False):
        val = torch.mean((random_output-predicted_output)**2)**0.5
        if log: 
            self.deki.append(val)
        normed_val = (val - self.mu) / (3*self.siggy+1e-6)
        
        return torch.tanh(normed_val)
    
    def update(self):
        self.mu = np.mean(self.deki)
        self.siggy = np.std(self.deki)  
        
    
    def training(self, random_output, predicted_out, dones, weights,updates,k):
        number_of_dones = torch.sum(dones)
        mse = torch.sum(weights*(1-dones)*(random_output-predicted_out)**2) / (dones.shape[0]) #-number_of_dones)
        return mse

class Normaliser:
    def __init__(self):
        self.deki_Vs = deque([],10000)
        self.Vs_sig_floor = 0.03
        self.Vs_sig = 0.01
    
    def update(self):
        self.Vs_sig = max(self.Vs_sig_floor, np.std(self.deki_Vs))