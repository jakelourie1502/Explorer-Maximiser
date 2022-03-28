# from msvcrt import kbhit
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
# from training_and_data.training import loss_func_RDN
sys.stdout = open("viewing_RDN_values.txt","w")

from global_settings import reward_support, lakes, goals, env_size, lake_coverage, actions_size, max_plays, dist, play_random, value_support, exp_v_support
from game_play.frozen_lake import gridWorld
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode

for j in range(1,10000):
    jake_zero = torch.load('../saved_models/jake_zero',map_location=torch.device('cpu'))

    pred = jake_zero.prediction
    dyn = jake_zero.dynamic
    repr =jake_zero.representation
    rdn_obj = jake_zero.rdn_obj
    jake_zero.eval()
    #1
    fut_nov_arr = np.zeros((48,))
    rdn_array = np.zeros((48,))
    batman_val_array = np.zeros((48,))
    robin_val_array = np.zeros((48,))
    reward_array = np.zeros((48,))
    
    print(jake_zero.rdn_obj.mu, jake_zero.rdn_obj.siggy)
    np.set_printoptions(suppress=True, precision = 2)

    if j % 2 ==0:
        print("Looking Two ahead")
    else:
        print("Looking one ahead)")
    for move in [0,1,2,3,4]:
        print("move: ", move)
        for i in range(48):
            dist = np.zeros((4*12+1))
            dist[i] = 1
            env=gridWorld(env_size,lakes, goals, n_actions = actions_size, max_steps = 8, dist = dist, seed = None, rnd=play_random,lake_cov=lake_coverage, randomly_create_env = False)
            raw_obs = env.reset()
            obs = vector2d_to_tensor_for_model(torch.tensor(raw_obs).float())
            s = jake_zero.representation(obs)
            s, r , t= jake_zero.dynamic(s,move)
            # # s, _ , t= jake_zero.dynamic(s,move)
            # # s, _ , t= jake_zero.dynamic(s,move)
            if j % 2 == 0:
                s, r , t= jake_zero.dynamic(s,move)
            _, p,v_rob, v_bat, n = jake_zero.prediction(s)
            r = support_to_scalar(r.detach().numpy(),*reward_support)
            v_rob = support_to_scalar(v_rob.detach().numpy(), *value_support)
            v_bat = support_to_scalar(v_bat.detach().numpy(), *value_support)
            t = torch.tensor(0)
            proj1 = jake_zero.RDN(s)
            proj2 = jake_zero.RDN_prediction(s)
            fut_nov = support_to_scalar(n.detach().numpy(), *exp_v_support)
            reward_array[i] = r
            r_nov = rdn_obj.evaluate(proj1, proj2)
            fut_nov_arr[i] =fut_nov*(1-t)
            rdn_array[i] = (1-t.detach().numpy())*r_nov.detach().numpy()
            batman_val_array[i] = v_bat
            robin_val_array[i] = v_rob
    

        print("Future nov array")
        print(fut_nov_arr.reshape(4,12))
        print("RDN value array instant")
        print(rdn_array.reshape(4,12))
        print('reward array')
        print(reward_array.reshape(4,12))
        sys.stdout.flush()
        # if j == 0:
        print("batman_value array")
        print(batman_val_array.reshape(4,12))
        print("robin_value array")
        print(robin_val_array.reshape(4,12))
    time.sleep(1*60)



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