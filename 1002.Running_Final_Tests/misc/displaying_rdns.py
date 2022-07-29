from cgi import test
from collections import deque
# from msvcrt import kbhit
import copy
import time
from pickle import FALSE
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from math import factorial as F
import matplotlib.pyplot as plt
from utils import support_to_scalar
import numpy as np
import torch
from preMadeEnvironments.Lakes6x6 import easy_version6x6, medium_version1_6x6, medium_version2_6x6, hard_version6x6
from preMadeEnvironments.Lakes8x8 import easy_version8x8, medium_version1_8x8, medium_version2_8x8, hard_version8x8
from utils import global_expl, Scores
from game_play.frozen_lake_different_each_time import gridWorld as Env
from game_play.play_episode import Episode
from game_play.mcts import MCTS, Node
from training_and_data.training import RDN_loss
from config import child
# sys.stdout = open("playing_game.txt","w")
class Ep_counter:
    def __init__(self):
        self.ids = []

from config import Config
cfg = Config()

from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
siam_scores = np.zeros((cfg.training.k)) + 0.5

class Ep_counter:
    def __init__(self):
        self.ids = []


ep_c = Ep_counter()
ExpMax = torch.load('../saved_models/jake_zeroWeak_pol_8x8Hard',map_location=torch.device('cpu'))

rdn_obj = RDN_loss()
rdn_obj.deki_stats = ExpMax.rdn_dekis
rdn_obj.new_ep_mu = 0

mu, sigma = rdn_obj.deki_stats['1'].mu, rdn_obj.deki_stats['1'].sigma
print(mu,sigma)
def leaky(x):
    s = np.sign(x-mu)
    
    x = ((x-mu)/(3*sigma)) * s
    
    return np.clip(-2,x,2)
    

move_count = 0
q_tracker = child()
q_tracker.end_states = 0
q_tracker.non_end_states = 0

cfg.env_map = hard_version8x8()

rdn_vals = np.zeros((cfg.env_size[0]*cfg.env_size[1]))
exp_v_vals = np.zeros((cfg.env_size[0]*cfg.env_size[1]))
with torch.no_grad():
    for i in range(0,cfg.env_size[0]*cfg.env_size[1]):
        expV_calc = 0
        expR = 0
        for j in range(cfg.actions_size):
            env=Env(cfg)
            env.starting_state = i
            
            obs_deque = deque([], cfg.deque_length)
            obs, _ = env.reset()
            obs_deque.append(obs)
            if cfg.store_prev_actions:
                action_frame = np.zeros_like(obs) + (4+1) / cfg.actions_size 
                obs_deque.append(action_frame)        
            obs_deque.append(obs)
            o = vector2d_to_tensor_for_model(np.array(obs_deque)).float()
            state = ExpMax.representation(o.float())
            state, _, _ = ExpMax.dynamic(state, j)
            r1 = ExpMax.RDN(state)
            r2 = ExpMax.RDN_prediction(state)
            _, _, expV = ExpMax.prediction(state, 0.17)
            expV = expV.detach().numpy() @ np.linspace(*cfg.prediction.expV_support)
            r_nov = -torch.nn.CosineSimilarity(dim=1)(r1,r2).detach().numpy()[0]
            r_nov = leaky(r_nov)
            
            expR += r_nov
            expV_calc += expV
        expR /= 5
        expV_calc /= 5

        rdn_vals[i] = int((expR)*100) / 100
        exp_v_vals[i] =int(expV_calc*100) / 100

    print(rdn_vals.reshape(cfg.env_size[0],cfg.env_size[1]))        
    print(exp_v_vals.reshape(cfg.env_size[0],cfg.env_size[1]))        
