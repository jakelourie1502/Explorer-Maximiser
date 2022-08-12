# from msvcrt import kbhit
from collections import deque

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
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function
import torch
# from training_and_data.training import loss_func_RDN
sys.stdout = open("viewing_RDN_values.txt","w")
from game_play.frozen_lake_different_each_time import gridWorld as Env
from config import Config
cfg = Config()
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode


print(np.array([y for x in env_gen_function(6,0.6) for y in x]).reshape(6,6))
cfg.env_map = np.array(
 [
 ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
 ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
 ['F', 'F', 'F', 'F', 'H', 'F', 'H', 'F'],
 ['F', 'H', 'H', 'F', 'F', 'F', 'F', 'F'],
 ['F', 'F', 'H', 'F', 'H', 'F', 'F', 'H'],
 ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'H'],
 ['F', 'H', 'F', 'H', 'H', 'F', 'F', 'H'],
 ['F', 'F', 'F', 'H', 'H', 'F', 'F', 'G']])
for j in range(1,10000):
    jake_zero = torch.load('../saved_models/jake_zero1',map_location=torch.device('cpu'))

    pred = jake_zero.prediction
    dyn = jake_zero.dynamic
    repr =jake_zero.representation
    jake_zero.eval()
    #1
    
    val_array = np.zeros((cfg.env_size[0]*cfg.env_size[1],))
    
    reward_array = np.zeros((cfg.env_size[0]*cfg.env_size[1],))
    term_array = np.zeros((cfg.env_size[0]*cfg.env_size[1],))
    
    
    np.set_printoptions(suppress=True, precision = 2)
    deki = deque([],cfg.deque_length)
    for _ in range(cfg.deque_length):
        deki.append(np.zeros((cfg.lake_size,cfg.lake_size)))
    
    if j % 2 ==0:
        print("Looking Two ahead")
    else:
        print("Looking one ahead)")
    for move in [0,1,2,3, 4]:
        print("move: ", move)
        for i in range(cfg.env_size[0]*cfg.env_size[1]):
            
                env=Env(cfg)
                env.starting_state = i
                
                raw_obs, _ = env.reset()
                
                dk = deki.copy()
                dk.append(raw_obs)
                dk = np.array(dk)
                obs = vector2d_to_tensor_for_model(torch.tensor(dk).float())
                
                s = jake_zero.representation(obs)
                s, r , t= jake_zero.dynamic(s,move)
                
                if j % 2 == 0:
                    s, r , t= jake_zero.dynamic(s,move)
                p, v = jake_zero.prediction(s)
                r = support_to_scalar(r.detach().numpy(),*cfg.dynamic.reward_support)
                v = support_to_scalar(v.detach().numpy(), *cfg.prediction.value_support)
                r1 = jake_zero.RDN(s)
                r2 = jake_zero.RND_prediction(s)
                t = torch.tensor(0)
                
                reward_array[i] = r
                term_array[i] = t
                
                val_array[i] = v

    

        print('reward array')
        print(reward_array.reshape(cfg.env_size[0],cfg.env_size[1]))
        print('terminal array')
        print(term_array.reshape(cfg.env_size[0],cfg.env_size[1]))
        sys.stdout.flush()
        # if j == 0:
        print("value array")
        print(val_array.reshape(cfg.env_size[0],cfg.env_size[1]))
        
    time.sleep(1*5)



models = [repr, dyn, pred]
epsilon = 0
temperature = 0.01
ep_h = []
def Play_Episode_Wrapper():
    with torch.no_grad():
        ep = Episode(jake_zero,epsilon)
        metrics, rew,_ = ep.play_episode(jake_zero.rdn_obj,False, epoch = 30, view_game =True)
        # replay_buffer.add_ep_log(metrics)
        print(rew)
        ep_h.append(rew)


def plot_beta(a,b):

    def normalisation_beta(a,b):
        return 1/ (F(a-1)*F(b-1) / F(a+b-1))

    def beta_val(a,b,val):
        return normalisation_beta(a,b) * val**(a-1) * (1-val)**(b-1)
    
    X = np.linspace(0,1,100)
    y = [beta_val(a,b,x) for x in X]
    plt.plot(X,y)

tn = time.time()
fig = plt.figure()
plt.savefig('saved_plt.png')

for _ in range(0):
    Play_Episode_Wrapper()
    a = int(1 + np.sum(ep_h))
    b = int(2 + len(ep_h) - np.sum(ep_h))
    plot_beta(a,b)
    plt.savefig('saved_plt.png')

    plt.pause(0.001)
    plt.close()


# print(np.sum(ep_h), np.mean(ep_h))
# print(time.time()- tn)