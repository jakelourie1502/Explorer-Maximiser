import copy
import numpy as np
# from .frozen_lake import gridWorld
from .frozen_lake import gridWorld as Env
import sys
sys.path.append("..")
sys.path.append(".")
from global_settings import gamma, bootstrap_start, exp_gamma, log_states,actions_size,batman_percentage
from global_settings import exp_v_support, env_size, N_steps, value_support
from utils import vector2d_to_tensor_for_model
from .mcts import Node, MCTS
import torch
import time
from collections import deque
 
class Episode:
    def __init__(self,model,epsilon,view_game=False):
        self.repr_model, self.dyn_model, self.prediction_model, self.rdn, self.rdn_pred = model.representation, model.dynamic, model.prediction, model.RDN, model.RDN_prediction
        self.epsilon = epsilon
        self.env=Env()
        self.gamma = gamma
        if np.random.uniform(0,1) < batman_percentage /2:
            self.actor = 'batman'
            self.actor_id = 0
        else:
            self.actor = 'robin'
            self.actor_id = 1
        
        if self.actor_id == 1 and np.random.uniform(0,1) < batman_percentage /2:
            Xs = np.linspace(0, self.env.max_steps, self.env.max_steps+1)
            midpoint = 0.5*(self.env.max_steps - 1)
            Ps = 1/(1+np.exp(-(Xs - midpoint)/np.sqrt(self.env.max_steps)))
            self.switch_prob = Ps[1:] - Ps[:-1]
            self.switch_prob /= np.sum(self.switch_prob)
            self.switch_step = np.random.choice(list(range(1,self.env.max_steps+1)),p=self.switch_prob)
        else:
            self.switch_step = self.env.max_steps+1 #will never be activated.

    def play_episode(self,RDN_OBJ, pick_best_policy=False, epoch=1,view_game = False):
        self.epoch = epoch
        if log_states: 
            exploration_logger = np.zeros((env_size[0]*env_size[1]+1)) 
        else:
            exploration_logger = []
        running_reward = 0
        analysis_log = []
        #### initialise a dictionary for for the episode to store things in.
        metrics = {}
        for met in ['action','obs','reward','done','policy','n_step_returns','v','exp_r', 'exp_v','actor']:
            metrics[met] = []
        
        obs = self.env.reset()
        metrics['obs'].append(obs) 
        mcts = MCTS(episode = self,epoch = epoch, pick_best = pick_best_policy,RDN_OBJ = RDN_OBJ)
        q_current = 1
        self.move_number=1
        while True:
            self.move_number +=1 
            if self.actor_id ==1:
                if self.move_number >= self.switch_step:
                    self.actor= 'batman'
                    self.actor_id = 0
                    mcts.reset_params()
                    

            if log_states: 
                exploration_logger[self.env.state] +=1
            
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.repr_model(obs.float())
            root_node = Node('null')
            root_node.state = state
            root_node.total_Q = q_current
            policy, action, root_node.total_Q, v, immediate_novelty, rootnode_v, analysis = mcts.one_turn(root_node) ## V AND rn_v is used for bootstrapping
            analysis[:,17] += self.env.n_steps
            analysis[:,18] += self.env.state
            analysis[:,19] += v
            analysis_log.append(analysis)
            q_current = root_node.total_Q
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.randint(0,actions_size)
            
            obs, _, reward, done = self.env.step(action)
            running_reward += reward
            self.store_metrics(action, reward, obs,metrics,done,policy, v, immediate_novelty, rootnode_v)
            if done == True:
                break 
        analysis = np.concatenate((analysis_log)) #7xstep_size, 21

        self.calc_reward(metrics) #using N step returns or whatever to calculate the returns.
        self.back_prop_unclaimed_novelty(metrics)
        n_step_return = metrics['n_step_returns']
        n_step_return = np.repeat(n_step_return,actions_size)
        
        analysis[:,20] = n_step_return
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        del obs
        
        return metrics, running_reward, exploration_logger, analysis
        
    def store_metrics(self, action, reward,obs, metrics,done,policy, v, immediate_novelty, exp_v):
        metrics['obs'].append(copy.deepcopy(obs))
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['done'].append(done)
        metrics['policy'].append(policy)
        metrics['v'].append(v)
        metrics['exp_r'].append(immediate_novelty)
        metrics['exp_v'].append(exp_v)
        metrics['actor'].append(self.actor_id)
    
    def calc_reward(self, metrics):
        Vs = metrics['v']
        Rs = np.array(metrics['reward'])
        Z = np.zeros_like((Rs))
        Z += Rs*gamma
        for n in range(1,N_steps):
            Z[:-n] += Rs[n:]*(gamma**(n+1))
        
        if self.epoch > bootstrap_start:
            Z[:-N_steps] += (gamma**(1*N_steps))*np.array(Vs[N_steps:]) #doubling the gamma exponenet to kill spurious value.
        
        Z = np.clip(Z,value_support[0],value_support[1])
        metrics['n_step_returns'] = Z

    def back_prop_unclaimed_novelty(self,metrics):
        n_steps_exp = N_steps*2
        #### because of how unstable exp_r is at the beginning, we start by just storing random numbers centred around for future_exp_val
        if self.epoch < bootstrap_start:
            Rs = np.array(metrics['exp_r'])
            Z = np.random.uniform(-0.1,0.1,Rs.shape[0])
        
        else:
            Vs = metrics['exp_v']
            Rs = np.array(metrics['exp_r'])
            Z = np.zeros_like((Rs)).astype('float')
            Z += Rs*exp_gamma
            for n in range(1,n_steps_exp):
                Z[:-n] += Rs[n:]*(exp_gamma**(n+1))
            Z[:-n_steps_exp] += (exp_gamma**(1*n_steps_exp))*np.array(Vs[n_steps_exp:]) 
            for i in range(1,min(N_steps+1,len(Vs))):
                Z[-i] += (gamma**i) * Vs[0]
            
        Z = np.clip(Z,exp_v_support[0]+1e-5,exp_v_support[1])
        metrics['future_exp_val'] = Z
        
        if np.random.uniform(0,256) < 1:
            print("Printing a sample from n step novelty returns, should be around 0 on average, going up to 9 max: ", Z)
            print("then printing the same exp r: ", metrics['exp_r'])
            sys.stdout.flush()
        