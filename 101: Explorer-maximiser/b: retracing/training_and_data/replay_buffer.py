import numpy as np
from global_settings import replay_buffer_size, batch_size, device
from global_settings import training_params
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
    def __init__(self):
        """'action','obs','reward','done','policy','n_step_returns','V'"""
        self.k = training_params['k']
        self.default_size = batch_size
        self.size = replay_buffer_size
        self.obs = []
        self.action_log = []
        self.reward_logs = []
        self.done_logs = []
        self.policy_logs = []
        self.n_step_returns = []
        self.predicted_v = []
        self.future_exp_val = []
        self.actor_id = []

    def add_ep_log(self, metrics):
        
        self.obs.extend(metrics['obs'])
        self.action_log.extend(metrics['action'])
        self.reward_logs.extend(metrics['reward'])
        self.done_logs.extend(metrics['done'])
        self.policy_logs.extend(metrics['policy'])
        self.n_step_returns.extend(metrics['n_step_returns'])
        self.predicted_v.extend(metrics['v'])
        self.future_exp_val.extend(metrics['future_exp_val'])
        self.actor_id.extend(metrics['actor'])

    def purge(self):
        no_of_examples = len(self.obs)
        if no_of_examples > self.size:
            reduc = no_of_examples - self.size
            self.obs = self.obs[reduc: ]
            self.action_log = self.action_log[reduc: ]
            self.reward_logs = self.reward_logs[reduc: ]
            self.done_logs = self.done_logs[reduc: ]
            self.policy_logs = self.policy_logs[reduc: ]
            self.n_step_returns = self.n_step_returns[reduc: ]
            self.predicted_v = self.predicted_v[reduc: ]
            self.future_exp_val = self.future_exp_val[reduc: ]
            self.actor_id = self.actor_id[reduc: ]
            
    def get_sample(self, prioritised_sampling = True, batch_size = batch_size):
        #### Need to add get sample prioritised.
        
        batch_n = batch_size
        if prioritised_sampling:
            
            coefs = torch.abs(torch.tensor(self.n_step_returns)-torch.tensor(self.predicted_v)) + 1e-3
            coefs = coefs[:-(self.k)]
            coefs = coefs / torch.sum(coefs)
            coefs = np.array(coefs)
            
            weights = (1/(coefs*len(self.predicted_v)))
            current_length = len(self.obs)-self.k #we don't want one right at the end or it will break.
            indices = np.random.choice(list(range(current_length)),size=batch_n, p=coefs,replace=True)
            weights = [weights[i] for i in indices]
            
        else:
            indices = np.random.randint(low = 0, high = len(self.obs)-self.k, size = batch_n)
            weights = np.ones_like(indices)
        sample_obs = np.array([self.obs[i] for i in indices])
        
        return torch.tensor(sample_obs).to(device), indices, weights
        
