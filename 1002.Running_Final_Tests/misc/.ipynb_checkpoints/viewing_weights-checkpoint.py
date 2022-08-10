from pickle import FALSE
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch



from global_settings import lakes, goals, env_size, lake_coverage, actions_size, max_plays, dist, play_random
from game_play.frozen_lake import gridWorld
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
jake_zero = torch.load('../saved_models/jake_zero',map_location=torch.device('cpu'))
pred = jake_zero.prediction_network
dyn = jake_zero.dynamic_network
repr =jake_zero.representation_network
model = torch.nn.Module()

def print_summary_stats(params):
    mean = torch.mean(params)
    mean_abs = torch.mean(torch.abs(params))
    param_variance = torch.var(params)
    print(f'mean: {mean} -- means_abs: {mean_abs} -- variance: {param_variance}')
    return len(params.flatten())

def view_model_stats(model):
    for param, name in zip(model.parameters(),model.named_parameters()):
        if 'weight' in name[0] or 'bias' in name[0]:
            print(name[0], ': ')
            print(print_summary_stats(param))

view_model_stats(repr)