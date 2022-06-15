from array import array
from logging import root
from re import A
import time
import numpy as np
from utils import support_to_scalar
import torch

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")
        

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        self.range = self.maximum - self.minimum
        

    def normalize(self, value, denom_only = False):
        if self.maximum > self.minimum:
            if denom_only:
                return (value) / ((self.maximum - self.minimum)+1e-3)
            else:# We normalize only when we have set the maximum and minimum values
                return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self,parent, p = 0, total_Q =0, Q =0 ,Qe = 0):
        self.Q_sum = 0
        self.Q = Q
        self.Qe_sum = 0
        self.Qe = Qe
        self.N = 0
        self.p = p
        self.parent = parent
        self.children = []
        self.exp_r = 0
        self.expV = 0
        self.r = 0
        self.v =0
        self.expTOT= 0 
        self.d = 0
        self.evaluated = False
        self.move = 0
        self.level = 0

class MCTS:
    def __init__(self, episode, epoch = 1, pick_best=False,view=False):
        self.view = view
        self.mm = MinMaxStats()
            #during testing we use pick_best = True, so temperatuer is irrelevant
        self.ep = episode
        self.cfg = self.ep.cfg
        self.epoch = epoch
        self.pick_best = pick_best
        self.c1, self.c2, self.gamma = self.cfg.mcts.c1,self.cfg.mcts.c2, self.cfg.gamma
        self.num_actions = self.cfg.actions_size
    
        self.set_temperature_and_sims()

    def set_temperature_and_sims(self):
        for key, val in self.cfg.mcts.temperature_changes.items():
            if self.epoch > key:
                self.temperature = val

        if self.cfg.mcts.manual_over_ride_play_limit == None:
            for key, val in self.cfg.mcts.sims.items():
                if self.epoch > key:
                    self.sims = val
        else:
             self.sims = self.cfg.mcts.manual_over_ride_play_limit
        
        if np.random.uniform(1,20000) < 3:
            print("Sims: ", self.sims, self.temperature, "epoch: ",self.epoch, "pick best: ",self.pick_best, "frame count: ", self.epoch)
    
    def one_turn(self,root_node,o):
        if self.cfg.mcts.norm_each_turn:
            self.mm = MinMaxStats()
        self.root_obs = o    
        self.root_node = root_node         
        idx = 0 
        for _ in range(self.sims):
            idx+=1
            self.mcts_go(root_node)
        
        #### This is purely for analytics
        if self.ep.epoch > 10000:
             for n in self.root_node.children:
                if n.r > 0.9 and n.Q > 0.95:
                    self.ep.q_tracker.end_states +=1
                elif n.Q > 0.9 and self.ep.move_number > 2:
                    self.ep.q_tracker.non_end_states+=1

        if self.pick_best:
            policy, chosen_action = self.pick_best_action(root_node)
        else:
            policy, chosen_action = self.randomly_sample_action(self.root_node)

        if self.view:
            print("Q: ", [x.Q for x in self.root_node.children])
            print("r: ", [x.r for x in self.root_node.children])
            print("done: ", [x.d for x in self.root_node.children])
            print("Qe: ", [x.Qe for x in self.root_node.children])
            print("Total Q:", [(x.Q + self.ep.rdn_beta*x.Qe) for x in self.root_node.children])
            print("exp_r: ", [float(x.exp_r) for x in self.root_node.children])
            print("expTOT: ", [float(x.expTOT) for x in self.root_node.children])
            print("policy: ", [float(x.p) for x in self.root_node.children])
            print("move: ", [x.move for x in self.root_node.children])
            for c in self.root_node.children:
                print("exp_r: ", [float(x.exp_r) for x in c.children])
        return policy, chosen_action, self.root_node.Q, self.root_node.v, self.root_node.Qe, self.root_node.children[chosen_action].exp_r
        
    def mcts_go(self,node):
        if not node.evaluated:
            self.expand(node)
        else:
            best_child = self.pick_child(node)
            self.mcts_go(best_child)
           
    def expand(self,node):
        """You've reached an unevaluated node
        First evalute it by calculating r and V and d, set evaluated flag to True
        then add unevaluated children
        """
        node.evaluated = True
        
        if node != self.root_node:
            ## GET TERMINAL INFO
            state, r, d = self.ep.dyn_model(node.parent.state,node.move) #
            d = d[0][0].detach().numpy()
            node.d = max(d,node.parent.d)
            node.state = state
            node.r = support_to_scalar(r[0].detach().numpy(),*self.cfg.dynamic.reward_support)
            
            
            ## Get r and exp_r
            if self.cfg.exploration_type != 'none':
                rdn_random = self.ep.model.RDN(state)
                rdn_real = self.ep.model.RDN_prediction(state)
                exp_r = self.ep.rdn_obj.evaluate(rdn_random, rdn_real, k = node.level)
                node.exp_r = ((1-node.parent.d) * exp_r).detach().numpy()
            else:
                node.exp_r = 0

        ## Get V and exp_V
        p, v, expV = self.ep.prediction_model(node.state,self.ep.rdn_beta)
        policy, v, expV = p[0], v[0], expV[0]
        v = support_to_scalar(v.detach().numpy(), *self.cfg.prediction.value_support)
        node.v = v
        

        if self.cfg.exploration_type == 'full' and self.epoch > self.cfg.start_training_expV:
            node.expV =support_to_scalar(expV.detach().numpy(),*self.cfg.prediction.expV_support)
            if node != self.root_node and self.cfg.mcts.model_expV_on_dones:
                
                node.expTOT = (1-node.d)*node.expV 
                
                if self.cfg.use_new_episode_expV:
                    node.expTOT += node.d * self.ep.rdn_obj.new_ep_mu 
                    
                if np.random.uniform(0,100000) < 1: print(f'unit test line 137 mCTS: want to be seeing expV values centred around 0, max(abs)=5, std hopefully 1: {node.expTOT}')
            else:
                node.expTOT = node.expV
                
        elif self.cfg.exploration_type != 'none':
            node.expTOT = node.exp_r
            if self.cfg.mcts.model_expV_on_dones:
                node.expTOT = (1-node.parent.d)*node.expTOT
        else:
            node.expTOT = 0
        

        if node!= self.root_node:
            self.back_prop_rewards(node, node.v/self.cfg.gamma, node.expTOT / self.cfg.exp_gamma) ##divide by gamma because we want to nullify the first discounting in backprop
        else:
            if self.ep.rdn_beta == 0: #maxi
                alpha = self.cfg.mcts.expl_noise_maxi.dirichlet_alpha
                noise_factor = self.cfg.mcts.expl_noise_maxi.noise
                pol_factor = 1-noise_factor
            else:
                alpha = self.cfg.mcts.expl_noise_explorer.dirichlet_alpha
                noise_factor = self.cfg.mcts.expl_noise_explorer.noise 
                pol_factor = 1 - noise_factor
            policy = pol_factor* policy
            dir = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha]*self.cfg.actions_size).float())
            sample_dir = dir.sample()
            policy += noise_factor*sample_dir
            self.root_node.N +=1
            

        ## Add a child node for each action of this node.
        for edge in range(self.num_actions):
            new_node = Node(parent=node, p=policy[edge], Q=node.Q,Qe=node.Qe)
            new_node.move = edge
            node.children.append(new_node)
            new_node.level = min(node.level + 1, self.cfg.training.k)
        
    

    def pick_child(self,node):
        Qs = np.array([x.Q for x in node.children]).reshape(-1,1) #5 x 1
        Qes = np.array([x.Qe for x in node.children]).reshape(-1,1) #5x 1
        Qps = np.array([x.p for x in node.children]).reshape(-1,1) #5 x 1
        Qns = np.array([x.N for x in node.children]).reshape(-1,1) #5 x 1
        
        if self.cfg.mcts.norm_Qs_OnMaxi and self.ep.rdn_beta == 0 and not self.cfg.mcts.norm_Qs_OnAll:
            Qs = self.mm.normalize(Qs)
            total_Qs = Qs
        
        elif self.cfg.mcts.norm_Qs_OnAll:
            total_Qs = self.mm.normalize(Qs+self.ep.rdn_beta*Qes)
        
        else:
            total_Qs = Qs + self.ep.rdn_beta * Qes

        if self.cfg.mcts.use_policy:
            policy_and_novelty_coef = Qps * np.sqrt(node.N) / (self.cfg.mcts.ucb_denom_k+Qns**self.cfg.mcts.exponent_node_n) #5x1
        else:
            policy_and_novelty_coef =  np.sqrt(1/2) * np.log(node.N + 1) / (1+ Qns)
        
        muZeroModerator = self.c1 + np.log((node.N + self.c2 + self.c1+1)/self.c2) #scalar
        
        child_val = (total_Qs + policy_and_novelty_coef * muZeroModerator).reshape(-1)
        return node.children[np.argmax(child_val)]   
        

    def back_prop_rewards(self, node, v,exp_bonus):
        """
        For real reward, we want to weight V by the future state we never reach with the current prediction for value
        """
        exp_bonus = exp_bonus*self.cfg.exp_gamma

        #in instant, this exp_v is just backpropagated. In full, we add the exp_r at each stage.
        if self.cfg.exploration_type == 'full'  and self.epoch > self.cfg.start_training_expV:
            exp_bonus += node.exp_r
            exp_bonus = np.clip(exp_bonus,*self.cfg.prediction.expV_support[:2])
        if np.random.uniform(0,200000) < 1: print(f'line 193 mcts: sample exp_bonus {exp_bonus}')
        v = v*self.cfg.gamma
        if not self.cfg.value_only:
            v = (1-node.d)*v
            v+= node.r
        else:
            v = (1-node.d)*v + node.d*node.v ###??? this isn't right. i'ts basically just v but i'm not using value only model
        node.N +=1

        node.Q_sum += v
        node.Qe_sum += exp_bonus
        node.Q = node.Q_sum / node.N
        node.Qe = node.Qe_sum / node.N
    
        if self.cfg.mcts.norm_Qs_OnAll:
            self.mm.update(node.Q + self.ep.rdn_beta * node.Qe)
        elif self.cfg.mcts.norm_Qs_OnMaxi:
            self.mm.update(node.Q)
                
        if node != self.root_node:
            self.back_prop_rewards(node.parent, v, exp_bonus)

    def randomly_sample_action(self,root_node):
        policy = np.array([float(x.N) ** (1 / self.temperature) for x in root_node.children])
        policy = policy / np.sum(policy)
        ##UNIT TEST
        if np.random.uniform(0,5000) < 1: print(f'UNIT TEST: sample policy line 217 mcts : {policy}')
        ##UNIT TEST
        return policy, np.random.choice(list(range(self.cfg.actions_size)), p=policy)

    def pick_best_action(self, root_node):
        policy = np.array([float(x.N) for x in root_node.children])
        policy = policy / np.sum(policy)
        
        return policy, np.argmax(policy)