import numpy as np
from config import Config
from numpy import save, load
import torch
class Replay_Buffer():

    """
    This is a class that can hold the data required for training
    each tuple is :
    obs_t = observed state at that time
    policy_log_t = policy after mcts process
    action_log_t = action chosen, which is a random.choice proprotional to policy.
    reward_log_t+1 = the reward achieved from Ot, At pair.
    done_log_t+1 = whether that Ot, At pair ended the game. note, in our game, reward =1 and done = True happens at the same time.
    fut_val_log_t = 
    """
    def __init__(self, cfg, training_flags):
        """'action','obs','reward','done','policy','n_step_returns','V'"""
        self.cfg = cfg
        self.training_flags = training_flags
        self.k = self.cfg.training.k
        self.default_size = self.cfg.training.batch_size
        self.size = self.cfg.training.all_time_buffer_size
        self.ep_id = []
        self.obs = []
        self.action_log = []
        self.reward_logs = []
        self.done_logs = []
        self.policy_logs = []
        self.n_step_returns = []
        self.n_step_returns_with_V = []
        self.predicted_v = []
        self.rdn_beta = []
        self.exp_r = []
        self.actor_id = []
        

    def add_ep_log(self, metrics):
        self.ep_id.extend(metrics['ep_id'])
        self.obs.extend(metrics['obs'])
        self.action_log.extend(metrics['action'])
        self.reward_logs.extend(metrics['reward'])
        self.done_logs.extend(metrics['done'])
        self.policy_logs.extend(metrics['policy'])
        self.n_step_returns.extend(metrics['n_step_returns'])
        self.predicted_v.extend(metrics['v'])
        self.rdn_beta.extend(metrics['rdn_beta'])
        self.exp_r.extend(metrics['exp_r'])
        self.n_step_returns_with_V.extend(metrics['n_step_returns_plusV'])
        self.actor_id.extend(metrics['actor_id'])

    def purge(self):
        no_of_examples = len(self.exp_r)
        if no_of_examples > self.size:
            reduc = no_of_examples - self.size
            self.ep_id = self.ep_id[reduc:]
            self.obs = self.obs[reduc: ]
            self.action_log = self.action_log[reduc: ]
            self.reward_logs = self.reward_logs[reduc: ]
            self.done_logs = self.done_logs[reduc: ]
            self.policy_logs = self.policy_logs[reduc: ]
            self.n_step_returns = self.n_step_returns[reduc: ]
            self.predicted_v = self.predicted_v[reduc: ]
            self.rdn_beta = self.rdn_beta[reduc: ]
            self.exp_r = self.exp_r[reduc: ]
            self.n_step_returns_with_V = self.n_step_returns_with_V[reduc:]
            self.actor_id = self.actor_id[reduc: ]

    def get_sample(self, prioritised_sampling = False, batch_size = None,exploration_sampling = False, resampling=False):
        #### Need to add get sample prioritised.
        if batch_size == None:
            batch_size = self.cfg.training.batch_size
        
        batch_n = batch_size
        min_len = min(len(self.n_step_returns),len(self.exp_r)) 
        if not resampling and not exploration_sampling: #this is if the length of the replay buffer varies for resampling
            start = max(0, min_len-self.cfg.training.replay_buffer_size)
        elif exploration_sampling:
            start = max(self.training_flags.expV_training_start_flag, min_len-self.cfg.training.replay_buffer_size_exploration)
            if np.random.uniform(0,100) < 1:
                print("start point for exploration sampling: ", start)
        else:
            start = 0
            
        end = min_len-2*self.k-self.cfg.N_steps_reward

        if prioritised_sampling:
            
            coefs = torch.abs(torch.tensor(self.n_step_returns_with_V[start:end])-torch.tensor(self.predicted_v[start:end])) #NEEDS TO BE TESTED WITH NSTEP_RETURNS + V
            coefs += 1e-3 
            current_length = len(coefs)
            coefs = coefs / torch.sum(coefs)
            coefs = np.array(coefs)
            
            weights = (1/(coefs*current_length))
            
            indices = np.random.choice(list(range(start,end)),size=batch_n, p=coefs,replace=True)
            weights = [weights[i-start] for i in indices]
            
        else:
            indices = np.random.randint(low = start, high = end, size = batch_n)
            weights = np.ones_like(indices)
        sample_obs = np.array([self.obs[i] for i in indices])
        
        return torch.tensor(sample_obs).to(self.cfg.device_train), indices, weights
        
    
    def save_to_disk(self):
        min_length = len(self.actor_id)
        np.savez('replay_buffer.npy', ep_id = self.ep_id[: min_length], obs = self.obs[: min_length], action = self.action_log[: min_length], reward=self.reward_logs[: min_length], 
                                            done=self.done_logs[: min_length], n_step = self.n_step_returns[: min_length],
                                            actor_id = self.actor_id[: min_length], rdn_beta = self.rdn_beta[: min_length], policy = self.policy_logs[: min_length],
                                            actor = self.actor_id[: min_length], expR = self.exp_r[: min_length],predicted_v = self.predicted_v[: min_length],
                                            epBuffer_ep_id = list(self.episodes.keys())[:-16], 
                                            epBuffer_len_metrics = [self.episodes[x]['length'] for x in list(self.episodes.keys())],
                                            epBuffer_obs = [self.episodes[x]['obs'] for x in list(self.episodes.keys())],
                                            epBuffer_actions = [self.episodes[x]['actions'] for x in list(self.episodes.keys())]
        )
    def load_from_disk(self):
        with load('replay_buffer.npy.npz', allow_pickle=True) as data:
            self.obs = list(data['obs'])
            self.action_log = list(data['action'])
            self.reward_logs = list(data['reward'])
            self.done_logs = list(data['done'])
            self.policy_logs = list(data['policy'])
            self.n_step_returns = list(data['n_step'])
            self.n_step_returns_with_V = self.n_step_returns
            self.predicted_v = list(data['predicted_v'])
            self.rdn_beta = list(data['rdn_beta'])
            self.exp_r = list(data['expR'])
            self.actor_id = list(data['actor'])
            
            self.epBufferobs = list(data['epBuffer_obs'])
            self.epBuffer_ep_id =list(data['epBuffer_ep_id'])
            self.epBuffer_len_metrics = list(data['epBuffer_len_metrics'])
            self.epBuffer_actions = list(data['epBuffer_actions'])
            self.episodes = {}
            for a, b, c, d in zip(self.epBuffer_ep_id, self.epBuffer_len_metrics, self.epBufferobs, self.epBuffer_actions):
                self.episodes[a] = {'length': b,
                                   'obs': list(c),
                                   'actions': list(d)}
                