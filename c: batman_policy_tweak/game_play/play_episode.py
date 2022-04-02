import copy
import numpy as np
# from .frozen_lake import gridWorld
from .skiing_env import skiWorld as Env
import sys
sys.path.append("..")
sys.path.append(".")
from global_settings import siam_floor as SF, gamma, bootstrap_start, exp_gamma, log_states,actions_size,batman_percentage, training_params
from global_settings import exp_v_support, env_size, N_steps, value_support
from utils import vector2d_to_tensor_for_model, Scores
from .mcts import Node, MCTS
import torch
import time
from collections import deque
 
class Episode:
    def __init__(self,model,epsilon,norman, siam_score=np.zeros((training_params['k']))+SF,actor_scores =Scores()):
        self.scores = actor_scores
        self.norman = norman
        if np.random.uniform(0,4096) < 1:
            print("Printing line 20 episode norman stats (vs_sig): ", self.norman.Vs_sig)
            print("line 20 episode, average scores (robin / batman / best): ", self.scores.robin_mu, " / ", self.scores.batman_mu, " / ", self.scores.best)
        self.model=model
        self.siam_floor = siam_score
        self.repr_model, self.dyn_model, self.prediction_model, self.rdn, self.rdn_pred = model.representation, model.dynamic, model.prediction, model.RDN, model.RDN_prediction
        self.epsilon = epsilon
        self.env=Env()
        self.gamma = gamma
        self.set_start_actor()
        self.set_switch_step()
        
        ##### initialise lists
        self.state_vectors = []
        self.move_number=0
        self.Vs = []
        self.V_unclaimed = []
        self.post_reward = []
        self.switched = False

    def set_start_actor(self):
        if np.random.uniform(0,1) < batman_percentage /2:
            self.actor = 'batman'
            self.actor_id = 0
        else:
            self.actor = 'robin'
            self.actor_id = 1

    def set_switch_step(self):
        if self.actor_id == 1 and np.random.uniform(0,1) < batman_percentage /2:
            Xs = np.linspace(0, self.env.max_steps, self.env.max_steps+1)
            midpoint = 0.5*(self.env.max_steps - 1)
            Ps = 1/(1+np.exp(-(Xs - midpoint)/np.sqrt(self.env.max_steps)))
            self.switch_prob = Ps[1:] - Ps[:-1]
            self.switch_prob /= np.sum(self.switch_prob)
            self.switch_step = np.random.choice(list(range(1,self.env.max_steps+1)),p=self.switch_prob)
        else:
            self.switch_step = self.env.max_steps+1 #will never be activated.

    def play_episode(self,RDN_OBJ, pick_best_policy=False, epoch=1):
        
        self.epoch = epoch
        if log_states: 
            exploration_logger = np.zeros((env_size[0]*env_size[1]+1)) 
        else:
            exploration_logger = []
        running_reward = 0
        
        #### initialise a dictionary for for the episode to store things in.
        metrics = {}
        for met in ['action','obs','reward','done','policy','n_step_returns','v','exp_r', 'exp_v','actor']:
            metrics[met] = []
        
        obs = self.env.reset()
        metrics['obs'].append(obs) 
        mcts = MCTS(episode = self,epoch = epoch, pick_best = pick_best_policy,RDN_OBJ = RDN_OBJ)
        total_Q = 1
        Q = 0.5
        Qe = 0.5
        
        while True:
            self.move_number +=1 
            if self.actor_id ==1:
                if self.move_number >= self.switch_step:
                    self.actor= 'batman'
                    self.actor_id = 0
                    mcts.reset_params()
                    self.switched = True

            if log_states: 
                exploration_logger[self.env.state] +=1
            
            obs = vector2d_to_tensor_for_model(obs) #need to double unsqueeze here to create fictional batch and channel dims
            state = self.repr_model(obs.float())

            ### Appending state vectors.
            proj = self.model.project(state,False)
            self.state_vectors.append(proj.clone())
            self.state_vectors_as_tensor = torch.cat(self.state_vectors)


            root_node = Node('null', total_Q=total_Q, Q=Q, Qe=Qe)
            root_node.state = state
            policy, action, total_Q, Q, Qe, v, immediate_novelty, rootnode_v = mcts.one_turn(root_node) ## V AND rn_v is used for bootstrapping
            
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.randint(0,actions_size)
            
            obs, _, reward, done = self.env.step(action)
            
            running_reward += reward
            reward = float(reward)
            if reward == 1:
                print("reward: ", self.actor)
            ### dealing with post reward stuff
            self.post_reward.append(0.)
            p_reward_array = np.array(self.post_reward)
            discount = np.flip(np.indices([len(self.post_reward)])).astype('float').reshape(-1)
            p_reward_array += reward * (gamma**discount)
            self.post_reward = list(p_reward_array)
            self.V_unclaimed = list(np.array(self.Vs) - p_reward_array)
            if np.random.uniform(0,20000) < 1:
                print("line 111 episode, taking a look at post reward, v unclaimed and v")
                print(self.post_reward, self.V_unclaimed, self.Vs)
            
            self.store_metrics(action, reward, obs,metrics,done,policy, v, immediate_novelty, rootnode_v)
            if done == True:
                break 

        self.calc_reward(metrics) #using N step returns or whatever to calculate the returns.
        self.back_prop_unclaimed_novelty(metrics)
        n_step_return = metrics['n_step_returns']
        n_step_return = np.repeat(n_step_return,actions_size)
        
        
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.
        if not self.switched:
            if self.actor == 'robin':
                self.scores.robin_deki.append(running_reward)
            else:
                self.scores.batman_deki.append(running_reward)
        del obs
        
        return metrics, running_reward, exploration_logger
        
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
        