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
from game_play.frozen_lakeGym_Image import gridWorld as Env
# from game_play.frozen_lake_KEY import gridWorld as Env
from game_play.play_episode import Episode
from game_play.mcts import MCTS, Node
from training_and_data.training import RDN_loss
from config import child
import argparse
# sys.stdout = open("playing_game.txt","w")
class Ep_counter:
    def __init__(self):
        self.ids = []

from config import Config
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', type=int, default=0)
parser.add_argument('--algo', type=int, default=0)
args = parser.parse_args()
cfg = Config(args.env, args.algo)

from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
siam_scores = np.zeros((cfg.training.k)) + 0.5

class Ep_counter:
    def __init__(self):
        self.ids = []


ep_c = Ep_counter()
ExpMax = torch.load('../saved_models/jake_zeroFA_FLKEY_1',map_location=torch.device('cpu'))

rdn_obj = RDN_loss(cfg)
rdn_obj.deki_stats = ExpMax.rdn_dekis
rdn_obj.new_ep_mu = 0
mu, sigma = rdn_obj.deki_stats['1'].mu, rdn_obj.deki_stats['1'].sigma

move_count = 0
q_tracker = child()
q_tracker.end_states = 0
q_tracker.non_end_states = 0

print(cfg.env_map)

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
            obs = (env.reset()).astype(np.float32)
            obs /= 255
            
            
            obs_deque.append(obs)
            if cfg.store_prev_actions:
                obs_deque.append(np.expand_dims(np.zeros_like(obs[0])+1,0))
                
            obs_deque.append(obs)
            
            o = vector2d_to_tensor_for_model(np.concatenate(obs_deque,0)).float()
            state = ExpMax.representation(o.float())
            state, _, _ = ExpMax.dynamic(state, j)
            r1 = ExpMax.RDN(state)
            r2 = ExpMax.RDN_prediction(state)
            _, _,_,_, expV = ExpMax.prediction(state, 1)
            expV = expV.detach().numpy() @ np.linspace(*cfg.prediction.expV_support)
            r_nov = rdn_obj.evaluate(r1,r2,1)
            expR += r_nov
            expV_calc += expV
        expR /= 5
        expV_calc /= 5

        rdn_vals[i] = expR
        exp_v_vals[i] =expV_calc

    plt.imshow(-rdn_vals.reshape(7,7), interpolation='nearest')
    plt.colorbar()
    plt.legend
    plt.show()
    plt.close()

    plt.imshow(-exp_v_vals.reshape(7,7), interpolation='nearest')
    plt.colorbar()
    plt.legend
    plt.show()
    plt.close()

    ### Initial map
    cfg.image_size = [192,192]
    env=Env(cfg)
    env.starting_state = 0
    obs_deque = deque([], cfg.deque_length)
    obs = (env.reset())
    plt.imshow(obs.transpose(1,2,0))
    plt.show()
    plt.close()
    ### N count
    heat_map = np.array(
        [
    [6465,  6465,  1500,   492,  1363, 1268,   691,   272,],
    [ 6465,  2653,   408,  1614,  1995,  1962,   601,    53],
    [ 4128,   696,   818,   989,   230,  1013,   354,    15],
    [ 3013,  1623,  1450,   612,    71,   464,    56,    23],
    [ 1538,   982,   517,    99,    36,   320,   155,   619],
    [ 1016,   582,   111,    49,   263,    37,    62,   522],
    [  919,   587,   120,   181,   329,    14,    11,   133],
    [ 1413,  1003,   833,  1118,   597,    29,     0,    25,]
        ]
    )
    heat_map = np.array([
  [    0,     0,     0,     0,     0,     0,     0],
  [    0,     0,     0,     0,     0,     0,     0],
  [   12,     0,     0,     0,     0,     0,     0],
  [    3,     6,     1,     0,    37,    44,    53],
  [    2,    17,    14,    72,  1174,  1555,  1473],
  [   28,   160,   349,   745,  3145,  3867,  3041],
  [   42,    40,    26,   117,  2415,  2904,  2906]
    ])
    heat_map = np.sqrt(heat_map/5 +0.01) + 1
    plt.imshow(heat_map, interpolation='nearest')
    plt.colorbar()
    plt.legend
    plt.show()
    plt.close()
    print(rdn_vals.reshape(cfg.env_size[0],cfg.env_size[1]))        
    print(exp_v_vals.reshape(cfg.env_size[0],cfg.env_size[1]))        
