import torch
import numpy as np
from config import Config
from collections import deque

cfg = Config()
class Scores:
    def __init__(self):
        self.rdns = np.round(np.linspace(*cfg.rdn_beta),2)
        self.scores = {}
        for b in self.rdns:
            self.scores[str(b)] = 0

    def update(self):
        pass
    
def vector2d_to_tensor_for_model(state):
    return torch.unsqueeze(torch.tensor(state),0)

def get_lr(training_step_counter):
    
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
    def __init__(self):
        self.log = np.zeros(cfg.exploration_logger_dims)
    def append(self, log):
        self.log += log