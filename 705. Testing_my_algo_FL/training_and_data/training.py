from numpy.core.defchararray import upper
import torch
import numpy as np 
from config import Config
from utils import scalar_to_support_batch
from collections import deque
from config import child
cfg = Config()
def loss_func_r(r, true_values, dones_k, weights):
    """
    r is an M by value_support[2]
    true_values is a M by 1
    weigths is a M by 1
    dones_k is an M by 1
    """
    
    true_values = scalar_to_support_batch(true_values, *cfg.dynamic.reward_support) #M by value_support[2]
    loss = -torch.sum(true_values * torch.log(r+1e-4),dim=1,keepdim=True)
    loss = weights * loss * (1-dones_k)
    return torch.mean(loss)

def loss_func_v(v, true_values, dones_k,weights):
    true_values = scalar_to_support_batch(true_values,*cfg.prediction.value_support)
    losses = -torch.sum(true_values * torch.log(v+1e-4),dim=1,keepdim=True)
    losses = weights*losses
    if not cfg.value_only:
        losses = losses * (1-dones_k)
    return torch.mean(losses)

def loss_func_expV(v, true_values, dones_k):
    true_values = scalar_to_support_batch(true_values,*cfg.prediction.expV_support)
    losses = -torch.sum(true_values * torch.log(v+1e-4),dim=1,keepdim=True)
    
    
    return torch.mean(losses)

def loss_func_p(p, true_policy, dones_k,weights):
    losses = torch.sum(true_policy * torch.log2(p + 1e-5),dim=1,keepdim=True)
    losses = -losses * weights * (1-dones_k)
    return torch.mean(losses)

def loss_func_entropy(p):
    return torch.mean(torch.sum(p * (torch.log2(p)+1e-3),dim=1))

def loss_func_proj(stopped_proj, w_grad_head, dones):
    #L1 loss
    contrastive_loss = torch.sum(stopped_proj*w_grad_head,dim=1,keepdims=True)/(
            (torch.sum(stopped_proj**2,dim=1,keepdims=True)*torch.sum(w_grad_head**2,dim=1,keepdims=True))**0.5)
    return -torch.mean((1-dones)*contrastive_loss)

def loss_func_future_nov(nov, true_unclaimed_nov, dones_k, weights):
    true_unclaimed_nov = scalar_to_support_batch(true_unclaimed_nov, *cfg.prediction.exp_v_support)
    losses = -torch.sum(true_unclaimed_nov * torch.log(nov+1e-4),dim=1,keepdim=True)
    losses = (1-dones_k)*weights*losses
    return torch.mean(losses)

class RDN_loss:
    def __init__(self):
        self.logs = 0
        self.a = 0.0001
        self.dekis = {}
        self.deki_stats = {}
        for c in range(cfg.training.k):
            self.deki_stats[str(c+1)] = child()
            self.dekis[str(c+1)] = deque([],10000)
            self.deki_stats[str(c+1)].mu = 0
            self.deki_stats[str(c+1)].sigma = 1
        self.update_log = []
        self.new_ep_expV_deki = deque([],10000)
        self.new_ep_mu = 0

    def tanchy(self,x):
        sng = torch.sign(x)
        absx = torch.abs(x)**2
        return 2*torch.tanh(1.5*absx*sng)

    def evaluate(self,rdn, pred,k):
        if cfg.RND_loss == 'cosine':
            val = -torch.nn.CosineSimilarity(dim=1)(rdn,pred)[0] #high cs -> v. negative (high aboslute neg)
        else:
            val = torch.sum((rdn-pred)**2)
        self.dekis[str(k)].append(val.numpy().item())

        normed_val = (val - self.deki_stats[str(k)].mu) / (3*self.deki_stats[str(k)].sigma + 1e-6) #high MSE -> high NOV value
        return self.tanchy(normed_val)
        
    def train(self, rdn, pred,dones, weights, k):
        if cfg.RND_loss == 'cosine':
            x = -torch.nn.CosineSimilarity(dim=1)(rdn,pred) #bn
        else:
            x = torch.sum((rdn-pred)**2,dim=1)
        
        x = x.reshape(-1,1) #bn, 1
        vals_for_expV = (1-dones)*(x-self.deki_stats[str(k)].mu) / (3*self.deki_stats[str(k)].sigma + 1e-6) #bn, 1 ---> having normed

        x = x * (1-dones) #right now we're removing those where last period was done.
        return self.tanchy(vals_for_expV), torch.mean(weights*x) #so we return the score we would get in the rdn_eval

    def update(self):
        for c in range(cfg.training.k):
            if len(self.dekis[str(c+1)]) > 0:
                self.deki_stats[str(c+1)].mu = np.mean(self.dekis[str(c+1)])
                self.deki_stats[str(c+1)].sigma = np.std(self.dekis[str(c+1)])
        self.new_ep_mu = np.mean(self.new_ep_expV_deki)
