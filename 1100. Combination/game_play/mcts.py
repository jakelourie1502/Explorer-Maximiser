from array import array
import copy
from logging import root
from re import A
import time
import numpy as np
from utils import support_to_scalar
import torch

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    Used for normalising in [0,1]
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")
        
    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        self.range = max(1e-3, self.maximum - self.minimum)
    
    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / max(1,(self.maximum - self.minimum))
        return value


class Node:
    def __init__(self,parent, p = 0, Q =0 ,Qep = 0, Qrnd = 0):
        self.Q_sum = 0
        self.Q = Q
        self.Qep_sum = 0
        self.Qep = Qep
        self.Qrnd_sum = 0
        self.Qrnd = Qrnd
        self.N = 0
        self.p = p #prior probability of taking an action
        self.parent = parent
        self.children = []
        self.r = 0
        self.v =0
        self.d = 0
        self.evaluated = False
        self.move = 0 #action taken from parent node to get to this node.
        self.level = 0 #this is the level in the tree, i.e. how many actions from the rootnode.
        self.exp_r = 0 #r^nov for rnd
        self.expV = 0 #v^nov 
        self.expTOT= 0  #used for backprop
        self.ep_nov = 0 #r^nov for episodic
        

class MCTS:
    def __init__(self, episode, epoch = 1, pick_best=False,view=False):
        self.view = view #used in manual playing, allows printing of Q and other values.
        self.mm = MinMaxStats() #initialises MMS for extrinsic reward each episode
        self.mm_rnd = MinMaxStats()
        self.ep = episode
        self.cfg = self.ep.cfg
        self.epoch = epoch
        self.pick_best = pick_best
        self.c1, self.c2, self.gamma = self.cfg.mcts.c1,self.cfg.mcts.c2, self.cfg.gamma
        self.num_actions = self.cfg.actions_size
        self.set_temperature_and_sims()

    def set_temperature_and_sims(self):
        ### This reads from cfg. Based on the global frame count (algo lifetime), sets temperature and sim count. 
        # In early runs, I use a low sim count typically, whilst the algo is learning.
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
    
    def one_turn(self,root_node):
        #Create a new MMS object each turn for the Q^total and Q^epinov
        self.mmtotal = MinMaxStats()
        self.mm_epi = MinMaxStats() #we create epi new each turn to slightly over-emphasise this when using 'both'.
        self.root_node = root_node         
        self.simmys = 0
        for _ in range(self.sims):
            self.mcts_go(root_node)
            self.simmys += 1
        
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
            print("rootnode Q: ", self.root_node.Q)
            print("highest value reached: ", self.mm.maximum)
            print("Q: ", [self.mm.normalize(x.Q) for x in self.root_node.children])
            print("r: ", [x.r for x in self.root_node.children])
            print("v: ", [x.v for x in self.root_node.children])
            print("done: ", [x.d for x in self.root_node.children])
            print("Qrnd: ", [x.Qrnd for x in self.root_node.children])
            print("normed Q rnds: ", [self.mm_rnd.normalize(x.Qrnd) for x in self.root_node.children])
            print("Qep: ", [x.Qep for x in self.root_node.children])
            Qeps = np.array([x.Qep for x in self.root_node.children]).reshape(-1,1) #5x 1
            normed_Qeps = (Qeps - np.min(Qeps) + self.mm_epi.range/self.num_actions) / (np.max(Qeps) - np.min(Qeps)+self.mm_epi.range/self.num_actions)
            print("normed Qeps: ", normed_Qeps)
            if self.cfg.both_exp_type == 'multiply':
                print("Total Q for episodic: ", [self.mmtotal.normalize(x.Q + \
                    self.ep.rdn_beta * (self.mm_epi.normalize(x.Qrnd)*normed_Qeps[y])) for x,y in zip(self.root_node.children,list(range(0,self.num_actions)))])
            
            print("exp_r: ", [float(x.exp_r) for x in self.root_node.children])
            print("expTOT: ", [float(x.expTOT) for x in self.root_node.children])
            print("epnov: ", [float(x.ep_nov) for x in self.root_node.children])
            print("policy: ", [float(x.p) for x in self.root_node.children])
            print("move: ", [x.move for x in self.root_node.children])
        ##### END OF ANALYTICS SECTION

        #to remain optimistic, we set the optimistic Q rnd and Qep to the best child of the chosen node.
        best_rnd_idx = np.argmax([x.Qrnd for x in self.root_node.children[chosen_action].children])
        best_Qep_idx = np.argmax([x.Qep for x in self.root_node.children[chosen_action].children])
        return policy, chosen_action, self.root_node.Q, self.root_node.v, \
                self.root_node.children[chosen_action].children[best_rnd_idx].Qrnd,\
                self.root_node.children[chosen_action].children[best_Qep_idx].Qep,\
                self.root_node.children[chosen_action].exp_r, self.root_node.children[chosen_action].r
        
    def mcts_go(self,node):
        ### In words...
        # if we reach a node we have already evaluated, explore best child (based on Q values).
        # Otherwise, evaluate the leaf node we reached
        if not node.evaluated:
            self.expand(node)
        else:
            best_child = self.pick_child(node)
            self.mcts_go(best_child)
           
    def expand(self,node):
        ### When we reach a leaf node, we evaluate it.
        node.evaluated = True #so we know not to evaluate it again.
        if node != self.root_node:
            ### next state, reward and terminal predictions (standard MuZero stuff)
            state, r, d = self.ep.model.dynamic(node.parent.state,node.move) #
            d = d[0][0].detach().numpy()
            node.d = max(d,node.parent.d)
            node.state = state
            node.r = support_to_scalar(r[0].detach().numpy(),*self.cfg.dynamic.reward_support)
            if self.cfg.train_dones:
                node.r = node.r * (1-node.parent.d)
            
            ## RND based novelty prediction
            if self.cfg.exploration_type in ['rdn','vNov_ablation','both']:
                rdn_random = self.ep.model.RDN(state)
                rdn_real = self.ep.model.RDN_prediction(state)
                node.exp_r = self.ep.rdn_obj.evaluate(rdn_random, rdn_real, k = node.level).detach().numpy()
                if self.cfg.mcts.model_expV_on_dones:
                    node.exp_r = (1-node.parent.d) * node.exp_r
            else:
                node.exp_r = 0

            ## Episodic novelty based calculation
            vec_cnt = len(self.ep.state_vectors)
            if self.cfg.exploration_type in ['episodic','both'] and vec_cnt > 5:
                ## We start episodic once we have at least 5 controllable states in memory
                ## Once memory buffer has 10 controllable states, we use 10 for comparison purposes.
                mn = min(vec_cnt, 10)
                
                # Calculate CosSim between controllable state and controllable states in memory buffer
                projection = self.ep.model.close_state_projection(node.state) # 1 x 128
                projections = projection.repeat(vec_cnt, 1) # moveNum, 128
                tensor_of_prev_visited_states = torch.cat(list(self.ep.state_vectors),0)
                neighbours = torch.nn.CosineSimilarity(dim=1)(projections, tensor_of_prev_visited_states).reshape(-1) #moveNum, 1 or movenum x 3 if distance
                gammas = torch.tensor(np.array([(0.5**(1/self.cfg.memory_size))**x for x in reversed(range(vec_cnt))]))
                neighbours *= gammas
                
                if self.view and node.parent == self.root_node:
                    print("move: ", node.move)
                    print('root node neighbours and sum of proj: ', neighbours)
                    
                    print("level: ", node.level)
                    print("length: ",len(tensor_of_prev_visited_states))
                    
                nearest_neighbours = neighbours.sort(descending=True)[0].numpy()
                nearest_neighbours = nearest_neighbours[:mn]
                node.ep_nov = np.sum((1-nearest_neighbours))*(100/mn)
                if node != self.root_node:
                    node.ep_nov = (1-node.parent.d) * node.ep_nov
                #UNIT TEST
                if np.random.uniform(0,2000) < 1: print("printing an ep nov before normalisation: ", node.ep_nov)
            else:
                node.ep_nov = 0
        
        ## Get V and exp_V
        node.v, policy, expV = self.extract_v_and_policy(node)
        
            
        if self.cfg.exploration_type in ['rdn','both'] and self.epoch > self.cfg.start_training_expV_min:
            ## In this branch, we are using RDN and we are also using v^nov
            node.expV =support_to_scalar(expV.detach().numpy(),*self.cfg.prediction.expV_support)
            if node != self.root_node and self.cfg.mcts.model_expV_on_dones:
                
                node.expTOT = (1-node.d)*node.expV 
                
                if self.cfg.use_new_episode_expV:
                    node.expTOT += node.d * self.ep.rdn_obj.new_ep_mu 
                    
                if np.random.uniform(0,100000) < 1: print(f'unit test line 137 mCTS: want to be seeing expV values centred around 0, max(abs)=5, std hopefully 1: {node.expTOT}')
            else:
                node.expTOT = node.expV

        elif self.cfg.exploration_type in ['rdn', 'both', 'vNov_ablation']: 
            #using rnd, but not v^{nov} (we only start using v^{nov} after ~20k frames)
            node.expTOT = node.exp_r
        else:
            #not using any measure of novelty.
            node.expTOT = 0
        

        if node!= self.root_node:
            self.back_prop_rewards(node, node.v/self.cfg.gamma, node.expTOT / self.cfg.exp_gamma, node.ep_nov / self.cfg.exp_gamma) ##divide by gamma because we want to nullify the first discounting in backprop
        else:
            if self.ep.actor_id != 0: 
                #Here, we create a base level of value to compare future Q values to.
                self.base_V = min(self.ep.current_best_score - self.ep.running_reward, support_to_scalar(node.v, *self.cfg.prediction.value_support))

            ### get the Q Qep and Qrnd for root node in the global stats
            self.mm.update((self.cfg.gamma **node.level)*node.Q)
            self.mm_epi.update(node.Qep) #we update global MM stats for Qep
            self.mm_rnd.update(node.Qrnd) #we update global MM stats for Qrnd
            #### For the rootnode:
            # We use a dirichlet alpha == [0.3, 0.5] to add some randomness to the probability priors for action selection as per original paper.
            if self.ep.actor_id == 0: #maxi not following other policy.
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
            new_node = Node(parent=node, p=policy[edge], Q=node.Q,Qep=node.Qep, Qrnd=node.Qrnd)
            new_node.move = edge
            node.children.append(new_node)
            new_node.level = min(node.level + 1, self.cfg.training.k)
            
    
    def extract_v_and_policy(self, node):
        if self.cfg.use_two_heads:
            p_maxi, p_expl, v_maxi, v_expl, expV = self.ep.model.prediction(node.state,self.ep.rdn_beta) #maxi is 0, so 1-0 is 1 == True
            p_maxi, p_expl, v_maxi, v_expl, expV = p_maxi[0], p_expl[0], v_maxi[0], v_expl[0], expV[0]
            v_expl = support_to_scalar(v_expl.detach().numpy(), *self.cfg.prediction.value_support)
            v_maxi = support_to_scalar(v_maxi.detach().numpy(), *self.cfg.prediction.value_support)
            v = max(v_expl, v_maxi)    
            
            if self.ep.actor_policy == 0:
                if np.random.uniform(0,20000) < 1:
                    print("rdn beta is 0 so we're just using the maxi policy")
                policy = p_maxi
            else:
                if np.random.uniform(0,10000) < 1:
                    print("using explorer policy with actor: ", self.ep.actor_id)
                policy = p_expl
            return v, policy, expV
        else:
            p, v, expV = self.ep.model.prediction(node.state,self.ep.rdn_beta)
            policy, v, expV = p[0], v[0], expV[0]
            v = support_to_scalar(v.detach().numpy(), *self.cfg.prediction.value_support)
            return v, policy, expV
        

    def pick_child(self,node):
        ### Create vectorised representation of all the nodes children.
        Qs = np.array([x.Q for x in node.children]).reshape(-1,1) #5 x 1
        Qrnds = np.array([x.Qrnd for x in node.children]).reshape(-1,1) #5x 1
        Qeps = np.array([x.Qep for x in node.children]).reshape(-1,1) #5x 1
        Qps = np.array([x.p for x in node.children]).reshape(-1,1) #5 x 1
        Qns = np.array([x.N for x in node.children]).reshape(-1,1) #5 x 1
        
        if self.ep.actor_id == 0: #maximiser just uses Q
            Qs = self.mm.normalize(Qs)
            total_Qs = Qs
        
        elif self.ep.actor_id == 1: #so we're in the explorer mode
            if self.cfg.VK:
                Qs = self.mm.normalize(Qs) - self.mm.normalize(self.base_V)
                Qs = np.where(Qs > 0, 1.5*Qs, 0.3 * Qs)
            else:
                Qs = self.mm.normalize(Qs) 
            
            if self.cfg.exploration_type in ['rdn', 'vNov_ablation']: #just using rnd
                total_Qs = self.mmtotal.normalize(Qs + self.ep.rdn_beta * Qrnds)
            
            if self.cfg.exploration_type == 'episodic':
                total_Qs = (self.ep.rdn_beta * self.mm_epi.normalize(Qeps)  + Qs)
            
            if self.cfg.exploration_type == 'both':
                ### First, normalise Qeps within that level
                normedQep = (Qeps - np.min(Qeps) + self.mm_epi.range/self.num_actions) / (np.max(Qeps) - np.min(Qeps)+self.mm_epi.range/self.num_actions)
                #then normalise RNDs globally by the episode
                normedQrnd = self.mm_rnd.normalize(Qrnds) 
                #Then we normalise on that turn the total Q so its [0,1]
                total_Qs = self.mmtotal.normalize(self.mm.normalize(Qs) + self.ep.rdn_beta * (normedQrnd*normedQep))
        
            if np.random.uniform(0,2000) < 1:
                print("stnum and move : ", self.ep.stnum, self.ep.move_number)
                print("level: ", node.level, "  normedQeps: ", normedQep, " normedQrnd: ", normedQrnd)
                print("Qs: ", Qs)
                print("novelty: ", self.ep.rdn_beta * (normedQrnd*normedQep))
                print("total Qs: ", total_Qs)
        
        if self.cfg.mcts.use_policy:
            policy_and_novelty_coef = Qps * np.sqrt(node.N) / (self.cfg.mcts.ucb_denom_k+Qns**self.cfg.mcts.exponent_node_n) #5x1
        else:
            policy_and_novelty_coef =  np.sqrt(1/2) * np.log(node.N + 1) / (1+ Qns)
        
        muZeroModerator = self.c1 + np.log((node.N + self.c2 + self.c1+1)/self.c2) #scalar
        
        child_val = (total_Qs + policy_and_novelty_coef * muZeroModerator).reshape(-1)

        return node.children[np.argmax(child_val)]   
        

    def back_prop_rewards(self, node, v,rnd_bonus, epi_bonus):
        """
        For real reward, we want to weight V by the future state we never reach with the current prediction for value
        """
        rnd_bonus = rnd_bonus*self.cfg.exp_gamma
        epi_bonus = epi_bonus*self.cfg.exp_gamma
        
        #exp_v is just backpropagated. In full, we add the exp_r at each stage.
        if self.cfg.exploration_type in ['rdn','both']  and self.epoch > self.cfg.start_training_expV_max:
            rnd_bonus += node.exp_r
            rnd_bonus = np.clip(rnd_bonus,*self.cfg.prediction.expV_support[:2])
        if np.random.uniform(0,200000) < 1: print(f'line 256 mcts: sample exp_bonus {rnd_bonus}')
        v = v*self.cfg.gamma
        if not self.cfg.value_only: #This is obsolete: currently haven't checked it works without using reward function, e.g. in chess games. so we also set not vlaue only.
            if self.cfg.train_dones:  
                v = (1-node.d)*v
            v+= node.r
        
        ### update node Q stats
        node.N +=1
        node.Q_sum += v
        node.Qrnd_sum += rnd_bonus
        node.Qep_sum += epi_bonus
        node.Q = node.Q_sum / node.N
        node.Qrnd = node.Qrnd_sum / node.N
        node.Qep = node.Qep_sum / node.N
    
        self.mm.update((self.cfg.gamma **node.level)*node.Q)
        self.mm_epi.update(node.Qep) #we update global MM stats for Qep
        self.mm_rnd.update(node.Qrnd) #we update global MM stats for Qrnd
        if self.cfg.exploration_type in ['rdn','vNov_ablation']:
            #we update total stats with normalized Q and un-normalised rdn_beta * Qrnd
            self.mmtotal.update(self.mm.normalize((self.cfg.gamma **node.level)*node.Q)+self.ep.rdn_beta*node.Qrnd) #for rnd
        
        elif self.cfg.exploration_type == 'both' and node !=self.root_node:
            
            Qeps = np.array([x.Qep for x in node.parent.children]).reshape(-1,1) #5x 1 get the Qeps of the siblings
            normedQep = (node.Qep - np.min(Qeps) + self.mm_epi.range/self.num_actions) / (np.max(Qeps) - np.min(Qeps)+self.mm_epi.range/self.num_actions) #normalise our particular nodes Qep [0,1]
            normedQrnd = self.mm_rnd.normalize(node.Qrnd)  #normalise global normedQrnd
            #so we have normed Qep in [0,1], normed Qrnd in [0,1]
            self.mmtotal.update(self.mm.normalize(node.Q) + self.ep.rdn_beta * (normedQrnd*normedQep)) #normalize Q and then update the total stats
             
            #On the first node, i.e. child of root, we want to also get the sibling stats and update the normaliser.
            #Otherwise, child node one might be way better but the normalisation total isn't updated so the gaps too big
            if self.simmys == 1 and node != self.root_node: 
                #unit test
                
                #
                for i in range(self.num_actions):
                    noddy = self.root_node.children[i]
                    normedQep = (noddy.Qep - np.min(Qeps) + self.mm_epi.range/self.num_actions) / (np.max(Qeps) - np.min(Qeps)+self.mm_epi.range/self.num_actions)
                    normedQrnd = self.mm_rnd.normalize(noddy.Qrnd) 
                    ## this bit is crucial, it creates norms which are used in UCB equation.
                    self.mmtotal.update(self.mm.normalize(noddy.Q) + self.ep.rdn_beta * (normedQep*normedQrnd))
                if np.random.uniform(0,100) < 1: 
                    print("unit test, this is the self.simmys==1 one. we should see that the node in question is at level 1 and simmy is 1: ", node.level, self.simmys)
                    

        if np.random.uniform(0,2000) < 1:
            print("in the backprop section, let's look at some average stats: ")
            print("epi: ", self.mm_epi.maximum, self.mm_epi.minimum, self.mm_epi.range)
            print("rnd: ", self.mm_rnd.maximum, self.mm_rnd.minimum)
            print("Q: ", self.mm.maximum, self.mm.minimum)
            print("totals: ", self.mmtotal.maximum, self.mmtotal.minimum)
                
        if node != self.root_node:
            self.back_prop_rewards(node.parent, v, rnd_bonus=rnd_bonus, epi_bonus=epi_bonus)

    def randomly_sample_action(self,root_node):
        policy = np.array([float(x.N) ** (1 / self.temperature) for x in root_node.children])
        policy = policy / np.sum(policy)
        ##UNIT TEST
        if np.random.uniform(0,100) < 1: print(f'UNIT TEST: sample policy line 217 mcts : {policy}')
        ##UNIT TEST
        return policy, np.random.choice(list(range(self.cfg.actions_size)), p=policy)

    def pick_best_action(self, root_node):
        policy = np.array([float(x.N) for x in root_node.children])
        policy = policy / np.sum(policy)
        
        return policy, np.argmax(policy)

    