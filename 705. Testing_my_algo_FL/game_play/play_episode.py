import copy
from shutil import move
import numpy as np
# from .frozen_lake import gridWorld

import sys
sys.path.append("..")
sys.path.append(".")
from utils import vector2d_to_tensor_for_model
from .mcts import Node, MCTS, MinMaxStats
from .Car_Driving_Env import RaceWorld as Env
import torch
from collections import deque
  
class Episode:
    def __init__(self,model,cfg, scores,ep_counter, epoch, rdn_obj,q_tracker, test_mode=False):
        self.test_mode = test_mode
        self.score_log = scores
        self.q_tracker = q_tracker
        self.rdn_obj = rdn_obj
        self.cfg = cfg
        if test_mode:
            self.rdn_beta = 0
            self.pick_best=True
        else:
            self.rdn_beta = np.random.choice(np.linspace(*self.cfg.rdn_beta))
            self.pick_best = False
        self.obs_deque = deque([], self.cfg.deque_length)
        self.epoch = epoch
        if ep_counter != False:
            self.ep_id = np.random.randint(0,1000000)
            while self.ep_id in ep_counter.ids:
                self.ep_id = np.random.randint(0,1000000)
            ep_counter.ids.append(self.ep_id)
        else: 
            ep_counter = 0
        
        self.model=model
        self.repr_model, self.dyn_model, self.prediction_model= model.representation, model.dynamic, model.prediction
        self.env= self.cfg.env(self.cfg)
        self.gamma = cfg.gamma
        
        ##### initialise lists
        self.state_vectors = []
        self.move_number=0
        

    def play_episode(self, view_game = False):
        
        if self.cfg.analysis.log_states: 
            exploration_logger = np.zeros(self.cfg.exploration_logger_dims) 
            
        else:
            exploration_logger = []
        self.running_reward = 0
        
        #### initialise a dictionary for for the episode to store things in.
        metrics = {}
        for met in ['ep_id', 'action','obs','reward','done','policy','n_step_returns','v','rdn_beta','exp_r']:
            metrics[met] = []
        
        obs   = self.env.reset()
        obs = obs.astype(np.float32) / 255
        
        for _ in range(self.cfg.deque_length // 2):
            self.obs_deque.append(np.zeros_like(obs))
            self.obs_deque.append(np.expand_dims(np.zeros_like(obs[0]),0))
        self.obs_deque.append(obs) #so this is now a list of 3, a (3, h ,w), a (1, h ,w) and a (3, h, w)
        metrics['obs'].append(copy.deepcopy(np.concatenate(self.obs_deque,0)))
        mcts = MCTS(episode = self,epoch = self.epoch, pick_best = self.pick_best)
        Q = 1
        Qe = 0
        
        while True:
            self.move_number+=1
            o = vector2d_to_tensor_for_model(np.concatenate(self.obs_deque, 0)) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.repr_model(o.float())

            root_node = Node('null', Q = Q, Qe = Qe)
            root_node.state = state
            policy, action, Q,  v, Qe, imm_nov = mcts.one_turn(root_node, o) ## V AND rn_v is used for bootstrapping
            if self.move_number == 1:
                first_move_Qe = Qe
            Qe = max(Qe,0) ## at the beginning, if expR is often negative, this compounds expV. so the first node selected will be much less negative after 1 sim.
            
            obs, reward, done, info  = self.env.step(action)
            obs = obs.astype(np.float32) / 255
            stnum = info['stnum']

            if self.cfg.analysis.log_states and not done: 
                exploration_logger[stnum[0],stnum[1]] +=1
            if view_game:
                print(action)
                print(obs)


            if self.cfg.store_prev_actions:
                action_frame = np.expand_dims(np.zeros_like(obs[0]),0) + (action+1) / self.cfg.actions_size 
                self.obs_deque.append(action_frame)
            self.obs_deque.append(obs)
            self.running_reward += reward
            
            reward = float(reward)
            if reward == 1:
                print("step number: ", self.move_number, "total reward: ", self.running_reward, " reward: ", reward, 'rdn_beta: ', self.rdn_beta)
                if not self.test_mode:
                     self.score_log.scores[str(np.round(self.rdn_beta,2))] += self.running_reward
            
            self.store_metrics(action, reward,metrics,done,policy, v, imm_nov)
            if done == True:
                break 
        
        self.calc_reward(metrics) #using N step returns or whatever to calculate the returns.
        
        
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        
        del obs
        
        return metrics, self.running_reward, exploration_logger, first_move_Qe
        
    def store_metrics(self, action, reward, metrics,done,policy, v,exp_r):
        metrics['ep_id'].append(self.ep_id)
        metrics['obs'].append(copy.deepcopy(np.concatenate(self.obs_deque,0)))
        
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['done'].append(done)
        metrics['policy'].append(policy)
        metrics['v'].append(v)
        metrics['rdn_beta'].append(self.rdn_beta)
        metrics['exp_r'].append(exp_r)
    
    def calc_reward(self, metrics):
        Vs = metrics['v']
        Rs = np.array(metrics['reward'])
        Z = np.zeros_like((Rs))
        Z += Rs*self.cfg.gamma
        for n in range(1,self.cfg.N_steps_reward):
            Z[:-n] += Rs[n:]*(self.gamma**(n+1))
        
        Z = np.clip(Z,self.cfg.prediction.value_support[0],self.cfg.prediction.value_support[1])
        metrics['n_step_returns'] = Z
        
        
        #this rmeoved because we're now using updated V* values.
        if self.epoch > self.cfg.calc_n_step_rewards_after_frames:
            Z_star = Z.copy()
            Z_star[:-self.cfg.N_steps_reward] += (self.cfg.gamma**(1*self.cfg.N_steps_reward))*np.array(Vs[self.cfg.N_steps_reward:]) #doubling the gamma exponenet to kill spurious value.
            metrics['n_step_returns_plusV'] = Z_star
        else:
            metrics['n_step_returns_plusV'] = Z

    