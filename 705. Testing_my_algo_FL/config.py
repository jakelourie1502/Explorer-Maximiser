import numpy as np
import torch
from game_play import *
import sys
import os
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function
from preMadeEnvironments.RaceWorld import small, medium, large
from preMadeEnvironments.Lake_Erroneous import erroneous
from preMadeEnvironments.Lakes6x6 import easy_version6x6, medium_version1_6x6, medium_version2_6x6, hard_version6x6
from preMadeEnvironments.Lakes8x8 import easy_version8x8, medium_version1_8x8, medium_version2_8x8, hard_version8x8
from game_play.Car_Driving_Env import RaceWorld
from game_play.frozen_lakeGym_Image import gridWorld
class child:
    def __init__(self):
        pass

class Config:
    def __init__(self):
        
        ##### From environment
        self.env = gridWorld
        self.same_env_each_time=True
        self.channels = 3
        self.env_size = [8,8]
        self.observable_size = [8,8]
        if self.same_env_each_time:
            #### TO BE EDITED FOR EACH MAP.
            self.env_map = medium_version2_8x8()
            self.max_steps = 100
        self.actions_size = 5

        
        #### END OF VERY ENVIRONMENT SPECIFIC STUFF.

        self.image_size = [48, 48]
        self.timesteps_in_obs = 2
        self.store_prev_actions = True
        if self.store_prev_actions:
            self.deque_length = self.timesteps_in_obs * 2 - 1
        else:
            self.deque_length = self.timesteps_in_obs
        

        

        # For all MLPs
        self.model = child()
        self.state_size = [6, 6]
        self.model.state_channels = 64
        self.model.res_block_kernel_size = 3
        
        # Representation
        self.repr = child()
        self.repr.conv1 = {'channels': 32,'kernel_size' : 3, 'stride':2, 'padding':1}
        self.repr.conv2 = {'channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        
        
        self.repr.res_block_channels = [32, 64] # , 64]
        self.repr.res_block_ds = [False, False] # True]

        # Dynamic
        self.dynamic = child()
        self.dynamic.conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.dynamic.res_blocks = [64,64] 
        self.dynamic.reward_conv_channels = 32
        self.dynamic.reward_hidden_dim = 64
        self.dynamic.reward_support = [-1,1,41]
        self.dynamic.terminal_conv_channels = 32
        self.dynamic.terminal_hidden_dim = 64
        
        #Prediction
        self.prediction = child()
        self.prediction.res_block = [64]
        self.prediction.value_conv_channels = 32
        self.prediction.value_hidden_dim = 64
        self.prediction.value_support = [-1,1,41]
        self.prediction.policy_conv_channels = 32
        self.prediction.policy_hidden_dim = 64
        self.prediction.expV_conv_channels = 32
        self.prediction.expV_hidden_dim = 64
        self.prediction.expV_support = [-10,10,51]

        ###RDN
        self.RND_output_vector = 256
        self.RND_loss = 'cosine' #cosine / MSE
        self.prediction_bias = True
        
        #### mcts functions
        self.mcts = child()
        self.mcts.c1 = 1
        self.mcts.c2 = 19652 
        # self.mcts.ucb_noise = [0,0.01]
        self.mcts.temperature_init = 1
        self.mcts.temperature_changes ={-1: 1, 3e6: 0.5 }
        self.mcts.sims = {-1:6,6000: 25}
        self.mcts.manual_over_ride_play_limit = None    #only used in final testing - set to None otherwise
        self.mcts.exponent_node_n = 1
        self.mcts.ucb_denom_k = 1
        self.mcts.use_policy = True
        self.mcts.expl_noise_maxi = child()
        self.mcts.expl_noise_maxi.dirichlet_alpha = .3
        self.mcts.expl_noise_maxi.noise = 0.3
        self.mcts.expl_noise_explorer = child()
        self.mcts.expl_noise_explorer.dirichlet_alpha = .5
        self.mcts.expl_noise_explorer.noise = 0.5
        self.mcts.model_expV_on_dones = False
        self.mcts.norm_Qs_OnMaxi = True
        self.mcts.norm_Qs_OnAll = False
        self.mcts.norm_each_turn = False
         
        #SimSiam
        self.use_siam = True
        self.siam = child()
        self.siam.negative_examples = False
        self.siam.project_rep = False
        self.siam.grad_rep = False
        self.siam.proj_l1 = 256
        self.siam.proj_out = 256
        self.siam.pred_hid = 128
        
        
        ### general algorithm functions
        #exploring
        self.exploration_type = 'none' #none / instant / full
        self.detach_expV_calc = True
        # self.rdn_beta = [0,.5,4]
        self.rdn_beta = [0,.0,1]
        self.value_only = False
        self.use_new_episode_expV = True
        self.start_training_expV_min = 10000
        self.start_training_expV = 30000 #can change.
        self.start_training_expV_siam_override = 0.8

        #General
        self.total_frames = 1000 * 1000
        self.update_play_model = 16
        self.gamma = 0.99
        self.exp_gamma = 0.95
        self.calc_n_step_rewards_after_frames = 10000
        self.N_steps_reward = 5
        self.start_frame_count = 0
        self.load_in_model = False
        self.analysis = child()
        self.analysis.log_states = True
        self.exploration_logger_dims = (1,self.env_size[0]*self.env_size[1])
        self.analysis.log_metrics = False
        self.device_train = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_selfplay = 'cpu'
        self.eval_x_frames = 10000
        self.eval_count = 25

        ### training
        self.training = child()
        self.training.replay_buffer_size = 50 * 1000
        self.training.replay_buffer_size_exploration = 1e6
        self.training.all_time_buffer_size = 1e7
        self.training.batch_size = 128
        self.training.play_workers = 12
        self.training.min_workers = 8
        self.training.max_workers = 32
        self.training.lr = 0.001
        self.training.lr_warmup = 1000+self.start_frame_count
        self.training.lr_decay = 1
        self.training.lr_decay_step = 100000
        self.training.optimizer = torch.optim.RMSprop
        self.training.momentum = 0.9
        self.training.l2 = 0.0001
        self.training.rho = 0.99
        self.training.k = 5
        self.training.coef = child()
        self.training.coef.value = 0.25
        self.training.coef.dones = 2.
        self.training.coef.siam = 2
        self.training.coef.rdn = 0.5
        self.training.coef.expV =0.5
        self.training.train_start_batch_multiple = 2
        self.training.prioritised_replay = True
        self.training.prioritised_replay_resampler = False
        self.training.resample_on = False
        self.training.resampling_start_frames = 10000
        self.training.ep_to_batch_ratio = [15,16]
        self.training.main_to_rdn_ratio = 2
        self.training.train_to_RS_ratio = 16
        self.training.on_policy_expV = False

        #assertions
        assert self.exploration_type in ['none', 'instant','full'], 'change exploration type to valid'
        if self.exploration_type == 'none':
            assert self.rdn_beta == [0,0,1], 'set rdn_beta to 0 if youre not exploring'
        assert self.RND_loss in ['cosine','MSE'], 'change RDN loss type'