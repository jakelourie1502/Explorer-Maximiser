from audioop import avg, lin2adpcm
from email.mime import base
from random import uniform
import torch
import numpy as np
from config import child
from collections import deque



def normalize_state_vec(encoded_state):
    min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
    max_encoded_state = (
        encoded_state.view(
            -1,
            encoded_state.shape[1],
            encoded_state.shape[2] * encoded_state.shape[3],
        )
        .max(2, keepdim=True)[0]
        .unsqueeze(-1)
    )
    scale_encoded_state = max_encoded_state - min_encoded_state
    scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
    return (
        encoded_state - min_encoded_state
    ) / scale_encoded_state

class Scores:
    def __init__(self,cfg):
        self.cfg = cfg
        self.rdns = np.round(np.linspace(*cfg.rdn_beta),3) + [0.0]
        self.scores = {}
        self.scores['maxi'] = child()
        self.scores['maxi'].log = deque([0], 500)
        self.scores['maxi'].ma = 0
        
        if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4':
            self.scores['maxi'].rooms_log = deque([0], 100)
            self.scores['maxi'].rooms_ma = deque([0], 100)
        for b in self.rdns:
            self.scores[str(b)] = child() 
            self.scores[str(b)].log = deque([0],500)
            self.scores[str(b)].ma = 0 
            self.scores[str(b)].adv = 0
            if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4':
                self.scores[str(b)].rooms_log = deque([0],100)
                self.scores[str(b)].rooms_ma = 0 

        self.probs = [1/cfg.rdn_beta[2]] * cfg.rdn_beta[2] 
        self.best_rdn_adv = np.round(cfg.rdn_beta[1],3)
        self.best_rdn_ma = np.round(cfg.rdn_beta[1],3)
        self.best_adv = 0
        self.maxi_score = 0
        self.last_test_scores =0. 
        self.best_actor = 0

    def update(self):
        self.maxi_score = self.scores['maxi'].ma = np.mean(self.scores[str('maxi')].log) + 1e-4
        best_adv = 0
        best_ma = 0
        sum_adv = 0
        best_rdn_adv = np.round(self.cfg.rdn_beta[1],3)
        best_rdn_ma = np.round(self.cfg.rdn_beta[1],3)
        if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4': 
            self.scores['maxi'].rooms_ma  = np.mean(self.scores['maxi'].rooms_log)

        baseline = max(self.maxi_score, self.last_test_scores)
        if np.random.uniform(0,20) < 1:
            print('maxi score, test score, baseline: ', self.maxi_score, self.last_test_scores, baseline)
        
        for b in self.rdns:
            scrs = np.array(self.scores[str(b)].log)
            self.scores[str(b)].ma = avg_score = np.mean(scrs) + 1e-4
            self.scores[str(b)].adv = advantage = np.mean(np.where(scrs - baseline > 0 , (scrs-baseline)**2, 0)) + 1e-4
            sum_adv+=advantage
            if advantage > best_adv:
                best_rdn_adv = b
                best_adv = advantage
            if avg_score > best_ma:
                best_rdn_ma = b
                best_ma = avg_score
            if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4': 
                self.scores[str(b)].ma = np.mean(np.array(self.scores[str(b)].log)+1e-4)
        
        self.best_rdn_adv = best_rdn_adv
        self.best_adv = best_adv - 1e-4
        
        self.best_rdn_ma = best_rdn_ma
        if best_ma > baseline:
            self.best_actor = 1
        else:
            self.best_actor = 0
        for idx, b in enumerate(self.rdns):
            self.probs[idx] = self.scores[str(b)].adv / sum_adv
        if np.random.uniform(0,20) < 1:
            print('maxi score, test score, baseline: ', self.maxi_score, self.last_test_scores, baseline)
            print("probs: ", self.probs)

            
def vector2d_to_tensor_for_model(state):
    return torch.unsqueeze(torch.tensor(state),0)

def get_lr(training_step_counter,cfg):
    
    if training_step_counter < cfg.training.lr_warmup:
        lr = cfg.training.lr * training_step_counter / cfg.training.lr_warmup
    else:
        lr = cfg.training.lr * cfg.training.lr_decay ** ((training_step_counter - cfg.training.lr_warmup) // cfg.training.lr_decay_step)
    return lr

def scalar_to_support(value, min_support, max_support, support_levels):
    """
    Note: takes numpy arrays
    in: requires scalar value, not array of one scalar value.
    outputs 1 dimensional vector
    """
    support_vals =  np.linspace(min_support, max_support, support_levels)
    support_vec = np.zeros((support_levels))
    if value in support_vals:
        idx = np.where(support_vals == value)
        support_vec[idx] = 1
    else:
        support_delim = (max_support-min_support) / (support_levels-1)

        lower_bound = (value // support_delim)*support_delim
        upper_bound = lower_bound+support_delim
        lower_proportion = round((upper_bound - value) / support_delim,5)
        upper_proportion = 1-lower_proportion
        lower_idx = int(round((lower_bound - min_support) / support_delim,1))
        upper_idx = lower_idx +1
        support_vec[lower_idx] = lower_proportion
        support_vec[upper_idx] = upper_proportion
    return support_vec

def support_to_scalar(array, min_support, max_support, support_levels):
    """
    in: 1d vector
    Note: takes numpy arrays
    outputs scalar
    """
    support_vals = np.linspace(min_support, max_support, support_levels)
    return round(np.sum(support_vals * array),4)

def scalar_to_support_batch(value, min_support, max_support, support_levels):
    
    support_delim = (max_support-min_support) / (support_levels-1) #the step value between support levels, e.g. if 0 -> 1, 11, then 0.1
    lower_bound = torch.floor(value / support_delim)*support_delim #ok so this takes a value like 0.15 and makes it 0.1
    upper_bound = lower_bound+support_delim
    lower_proportion = torch.round(10**3*((upper_bound - value) / support_delim)) / 10**3
    upper_proportion = 1-lower_proportion

    lower_idx = torch.round((lower_bound - min_support) / support_delim).long().squeeze()
    upper_idx = torch.where(lower_idx == support_levels-1,0,lower_idx + 1)
    try:
        lower_idx = torch.nn.functional.one_hot(lower_idx,support_levels)*lower_proportion
    except:
        print("lower bound issue: ", min_support, lower_idx, value, lower_bound)
        lower_bound = 0
        lower_idx = torch.nn.functional.one_hot(lower_idx,support_levels)*lower_proportion
        upper_idx = torch.where(lower_idx == support_levels-1,0,lower_idx + 1)
    upper_idx = torch.nn.functional.one_hot(upper_idx,support_levels)*upper_proportion
    support_vec = lower_idx+upper_idx


    return support_vec

class global_expl:
    def __init__(self,cfg):
        self.log = np.zeros(cfg.exploration_logger_dims)
    def append(self, log):
        self.log += log