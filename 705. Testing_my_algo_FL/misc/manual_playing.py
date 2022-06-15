import matplotlib.pyplot as plt
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
# import matplotlib.pyplot as plt
from utils import support_to_scalar
import numpy as np
import torch
from preMadeEnvironments.Lakes6x6 import easy_version6x6, medium_version1_6x6, medium_version2_6x6, hard_version6x6
from preMadeEnvironments.Lakes8x8 import easy_version8x8, medium_version1_8x8, medium_version2_8x8, hard_version8x8
from preMadeEnvironments.Lake_Erroneous import erroneous
from utils import global_expl, Scores
from game_play.frozen_lakeGym_Image import gridWorld as Env
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
ExpMax = torch.load('../saved_models/jake_zeroTestingAvgPool',map_location=torch.device('cpu'))

rdn_obj = RDN_loss()
rdn_obj.deki_stats = ExpMax.rdn_dekis
rdn_obj.new_ep_mu = 0

move_count = 0
q_tracker = child()
q_tracker.end_states = 0
q_tracker.non_end_states = 0

cfg.env_map = erroneous()



for _ in range(5):
    with torch.no_grad():
        ep = Episode(ExpMax, cfg, Scores(), Ep_counter(),1e6, rdn_obj = rdn_obj, test_mode=False, q_tracker=q_tracker)
        ep.rdn_beta = 0.33
        mcts  = MCTS(ep, epoch = 1e6, pick_best=False,view=True)
        env = Env(cfg)
        
        obs_deque = deque([], cfg.deque_length)
        done = False
        obs= env.reset()
        obs = obs.astype(np.float32) / 255
        for _ in range(cfg.deque_length // 2):
            obs_deque.append(np.zeros_like(obs))
            obs_deque.append(np.expand_dims(np.zeros_like(obs[0]),0))
        obs_deque.append(obs) #so this is now a list of 3, a (3, h ,w), a (1, h ,w) and a (3, h, w)
        print(len(obs_deque))
        Q = 1
        Qe = 0.5

        while not done:
            
            
            move_count +=1
            
            o = vector2d_to_tensor_for_model(np.concatenate(obs_deque,0)) #need to double unsqueeze here to create fictional batch and channel dims
            state = ExpMax.representation(o.float())
            root_node = Node('null', Q = Q, Qe = Qe)
            root_node.state = state
            policy, action, Q,  v, Qe, _ = mcts.one_turn(root_node, o) ## V AND rn_v is used for bootstrapping
            print(policy)
            print("take this action: ", action)
            plt.close()
            plt.imshow(obs.transpose(1,2,0))
            plt.show(block=False)
            Qe = max(Qe,0) ## at the beginning, if expR is often negative, this compounds expV. so the first node selected will be much less negative after 1 sim.
            
            act = int(input("Give it to me"))
            obs, reward, done, stnum  = env.step(act)
            obs = obs.astype(np.float32) / 255
            if cfg.store_prev_actions:
                action_frame = np.expand_dims(np.zeros_like(obs[0]),0) + (action+1) / cfg.actions_size 
                obs_deque.append(action_frame)
            obs_deque.append(obs)
            
            # print(obs_deque)
            rdn_vals = []
            # for a in range(5):
            #     s_copy, r, _ = ExpMax.dynamic(s.clone(), a)
            #     r1 = ExpMax.RDN(s_copy)
            #     r2 = ExpMax.RDN_prediction(s_copy)

                
            #     # print(r)
                
            #     p, v, n = ExpMax.prediction(s_copy,0.)
                
            #     v = support_to_scalar(v.detach().numpy(), *cfg.prediction.value_support)
            #     r = support_to_scalar(r.detach().numpy(), *cfg.dynamic.reward_support)
            #     n = support_to_scalar(n.detach().numpy(),*cfg.prediction.expV_support)
                
            #     print(r,  v ,n, (rdn_obj.evaluate(r1,r2)),torch.sqrt(torch.sum(torch.abs(r1))))
            #     r,v = ExpMax.aux(o.float(),a,1)
            #     v = support_to_scalar(v.detach().numpy(), *cfg.prediction.value_support)
            #     r = support_to_scalar(r.detach().numpy(), *cfg.dynamic.reward_support)
            #     # print(r,  v)
            #     # r1 = ExpMax.RDN(s_copy_2)
            #     # r2 = ExpMax.RDN_prediction(s_copy_2)
                
            