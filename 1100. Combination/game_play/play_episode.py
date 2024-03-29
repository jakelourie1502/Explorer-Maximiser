import copy
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
from utils import vector2d_to_tensor_for_model
from collections import deque
from .mcts import Node, MCTS
from .wrappers import make_atari
import cv2

class Episode:
    def __init__(self,model,cfg, scores,ep_counter, epoch, rdn_obj,q_tracker, test_mode=False, current_best_score = 0):
        self.current_best_score = current_best_score
        self.test_mode = test_mode
        self.score_log = scores
        self.q_tracker = q_tracker
        self.rdn_obj = rdn_obj
        self.cfg = cfg
        self.use_two_heads = self.cfg.use_two_heads
        self.select_actor()
        self.model=model

        ### This creates a unique id for the episode, used when doing re-analyse
        if ep_counter != False:
            self.ep_id = np.random.randint(0,1000000)
            while self.ep_id in ep_counter.ids:
                self.ep_id = np.random.randint(0,1000000)
            ep_counter.ids.append(self.ep_id)
        else: 
            ep_counter = 0
        
        self.epoch = epoch #epoch == frame_count
        if self.cfg.atari_env:
            self.env= make_atari(self.cfg.env)
        else:
            self.env= self.cfg.env(self.cfg)
        self.gamma = cfg.gamma
        
        ##### initialise lists and running totals
        self.obs_deque = deque([], self.cfg.deque_length)
        self.move_number=0
        self.running_reward = 0
        self.state_vectors = deque([],self.cfg.memory_size)

    def play_episode(self):
        exploration_logger = np.zeros(self.cfg.exploration_logger_dims) if self.cfg.analysis.log_states else [] #used to log true state count
        
        #### initialise a dictionary for for the episode to store things in.
        metrics = {}
        for met in ['ep_id', 'action','obs','reward','done','policy','n_step_returns','v','rdn_beta','exp_r','actor_id','stnum']:  metrics[met] = []
        
        # Create environment and get into correct format
        if self.cfg.atari_env:
            obs = self.env.reset()
            obs = cv2.resize(obs, (self.cfg.image_size[0],self.cfg.image_size[1]), interpolation=cv2.INTER_AREA)
            obs = np.transpose((obs.astype(np.float32) / 255),(2,0,1))
        else:
            obs  = self.env.reset()
            obs = obs.astype(np.float32) / 255
         
        #Add obs to deque (for first observation, add array of 0s)
        for _ in range(self.cfg.deque_length // 2):
            self.obs_deque.append(np.zeros_like(obs))
            self.obs_deque.append(np.expand_dims(np.zeros_like(obs[0]),0))
        self.obs_deque.append(obs) #so this is now a list of 3, a (3, h ,w), a (1, h ,w) and a (3, h, w)
        
        metrics['obs'].append(copy.deepcopy(np.concatenate(self.obs_deque,0))) #store first obs in metrics
        mcts = MCTS(episode = self,epoch = self.epoch, pick_best = self.pick_best) #initialise MCTS
        
        #### We use optimistic Q value estimates to kick off to encourage exploration
        Q = 1
        Qep = 100 if self.cfg.exploration_type in ['episodic','both'] else 0 #max possible score
        Qrnd = 0
        self.stnum = 0 #for unit testing in mcts
        while True:
            self.move_number+=1 #for counting purposes

            #### Get obs in correct form and extract hidden state
            o = vector2d_to_tensor_for_model(np.concatenate(self.obs_deque, 0)) 
            state = self.model.representation(o.float())

            #### Append controllable state to memory if using episodic novelty
            if self.cfg.exploration_type in ['episodic','both']:  self.state_vectors.append(copy.deepcopy(self.model.close_state_projection(state)))
            
            #### Initialise root node and run MCTS
            root_node = Node('null', Q = Q, Qrnd = Qrnd, Qep = Qep)
            root_node.state = state
            policy, action, Q,  v, Qrnd, Qep, imm_nov, expected_reward = mcts.one_turn(root_node) ## V AND rn_v is used for bootstrapping
            
            a = action+8 if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4' and action in [6,7] else action
            
            if self.move_number == 1: first_move_Qe = Qrnd #used for calculating new ep nov for RND novelty.
            Qrnd = max(Qrnd,0) ## at the beginning, if expR is often negative, this compounds expV. so the first node selected will be much less negative after 1 sim.
            Qep = max(Qep, 0)
            
            ### Play move, log reward and create observation in right form
            obs, reward, done, info  = self.env.step(a)
            reward = float(reward)
            if self.cfg.reward_clipping: reward = np.clip(reward,-1,1)
            self.running_reward += reward
            if self.cfg.atari_env:
                obs = cv2.resize(obs, (self.cfg.image_size[0],self.cfg.image_size[1]), interpolation=cv2.INTER_AREA)
                obs = np.transpose((obs.astype(np.float32) / 255),(2,0,1))
            else:
                obs = obs.astype(np.float32) / 255
            
            ### Update the deque
            if self.cfg.store_prev_actions:
                action_frame = np.expand_dims(np.zeros_like(obs[0]),0) + (action+1) / self.cfg.actions_size 
                self.obs_deque.append(action_frame)
            self.obs_deque.append(obs)
            
            ### Logging, printing for unit tests and storing metrics
            stnum = 0 if self.cfg.atari_env else info['stnum']
            self.store_metrics(action, reward,metrics,done,policy, v, imm_nov,stnum)
            if self.cfg.analysis.log_states and not done: 
                exploration_logger[stnum[0],stnum[1]] +=1
            if reward >= 0.01:
                print("actor: ", self.actor_id, "policy actor: ", self.actor_policy, " step number: ", self.move_number, "total reward: ", self.running_reward, " reward: ", reward, 'rdn_beta: ', self.rdn_beta)            
            if done == True:
                break 
        
        #End of game - log scores, calculate N-step returns.
        if self.actor_id == 1:
            self.score_log.scores[str(np.round(self.rdn_beta,3))].log.append(self.running_reward)
        else:
            self.score_log.scores['maxi'].log.append(self.running_reward)
        self.calc_reward(metrics) #using N step returns or whatever to calculate the returns.
        metrics['obs'] = metrics['obs'][:-1] #otherwise we'd have one extra observation.

        ##Montezuma bit
        if self.cfg.env == "MontezumaRevengeNoFrameskip-v4": self.log_for_montezuma(info) 
        return metrics, self.running_reward, exploration_logger, first_move_Qe
        
    def log_for_montezuma(self, info):
        rooms_visited = len(info['episode']['visited_room'])
        print("Actor: ", self.actor_id, "rooms visited: ", info['episode']['visited_room']) 
        if self.actor_id == 1:
            self.score_log.scores[str(np.round(self.rdn_beta,3))].rooms_log.append(rooms_visited)
        else:
            self.score_log.scores['maxi'].rooms_log.append(rooms_visited)

    def store_metrics(self, action, reward, metrics,done,policy, v,exp_r,stnum):
        metrics['ep_id'].append(self.ep_id)
        metrics['obs'].append(copy.deepcopy(np.concatenate(self.obs_deque,0)))
        metrics['stnum'].append(stnum)
        metrics['action'].append(action)
        metrics['reward'].append(reward)
        metrics['done'].append(done)
        metrics['policy'].append(policy)
        metrics['v'].append(v)
        metrics['rdn_beta'].append(self.rdn_beta)
        metrics['exp_r'].append(exp_r)
        metrics['actor_id'].append(self.actor_id)

    def calc_reward(self, metrics):
        ### This calculates N step returns for extrinsic reward.
        Rs = np.array(metrics['reward'])
        Z = np.zeros_like((Rs))
        Z += Rs*self.cfg.gamma
        for n in range(1,self.cfg.N_steps_reward):
            Z[:-n] += Rs[n:]*(self.gamma**(n+1))
        
        Z = np.clip(Z,self.cfg.prediction.value_support[0],self.cfg.prediction.value_support[1])
        metrics['n_step_returns'] = Z
        metrics['n_step_returns_plusV'] = Z

    def select_actor(self):
        if self.cfg.use_two_heads:
            if self.test_mode: #when test mode, we want maximiser; selecting most played action and the best actor policy based on scores
                self.actor_id = 0 #maxi
                self.pick_best = True
                self.rdn_beta = self.score_log.best_rdn_ma
                if self.score_log.best_actor == 0:
                    self.actor_policy = 0
                else:
                    self.actor_policy = 1

            elif (np.random.uniform(0,1) < self.cfg.explorer_percentage): #if using explorer, we want explorer actor and actor policy and randomly select RND beta
                self.actor_id = 1 #explorer
                self.rdn_beta = np.round(np.random.choice(np.linspace(*self.cfg.rdn_beta),p=self.score_log.probs),3)
                self.pick_best = False
                self.actor_policy = 1
        
            else: #this is maximiser playing a non-evaluation run; we select a rnd beta to utilise explorer and maximiser value funcs.
                self.actor_id = 0
                self.pick_best = False
                self.rdn_beta = np.round(np.random.choice(np.linspace(*self.cfg.rdn_beta),p=self.score_log.probs),3)
                if self.score_log.best_adv > 0 and np.random.uniform(0,1) < self.cfg.follow_better_policy:
                    #if the explorer is doing better than the maximiser, we may want to follow its policy.
                    self.actor_policy = 1
                    if np.random.uniform(0,5) < 1:
                        print("using another actor")
                else:
                    self.actor_policy = 0
                if np.random.uniform(0,5) < 1:
                    print("from probs: ", (self.score_log.probs))
        

        elif self.cfg.exploration_type != 'none': 
            #so not using two heads but using some exploration
            if self.test_mode: #maxi ; evaluation run.
                self.actor_id = 0 #maxi
                self.pick_best = True
                if self.cfg.explorer_percentage == 1:
                    self.rdn_beta = self.cfg.rdn_beta[0] #for the one where we only use one thing.
                else:
                    self.rdn_beta = 0.0
                
            elif (np.random.uniform(0,1) < self.cfg.explorer_percentage and not self.resampling): #using explorer
                self.actor_id = 1 #explorer
                self.rdn_beta = np.round(np.random.choice(np.linspace(*self.cfg.rdn_beta)),3)
                self.pick_best = False
                
            else: #using maximiser, not evaluation run.
                self.actor_id = 0
                self.pick_best = False
                self.rdn_beta = 0.0
            self.actor_policy = self.actor_id
        else: #this is for MuZeroSS or vanilla MuZero.
            self.actor_policy = 0
            self.actor_id = 0 #maxi
            self.rdn_beta = 0.0
            self.pick_best = self.test_mode