import numpy as np
import torch
from game_play import *
import sys
import os
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function
### We import all the environments and env. maps
# we then use a present --env X to specify which environment we want to use.
from preMadeEnvironments.RaceWorld import small, medium, large
from preMadeEnvironments.Lake_Erroneous import erroneous, erroneous_with_second_goal
from preMadeEnvironments.Lakes6x6 import easy_version4x4, easy_version5x5, easy_version6x6, medium_version1_6x6, medium_version2_6x6, hard_version6x6
from preMadeEnvironments.Lakes8x8 import easy_version8x8, medium_version1_8x8, medium_version2_8x8, hard_version8x8
from preMadeEnvironments.Key_Envs import key1
from game_play.Car_Driving_Env import RaceWorld
from game_play.frozen_lake_KEY import gridWorld as keyWorld
from game_play.frozen_lakeGym_Image import gridWorld

#This allows the formulation of e.g. cfg.training.batch_size
class child:
    def __init__(self):
        pass

#### ENUMS FOR ALGORITHMS
NO_PRESET = 0
FULL_ALGO_RND = 1
FULL_ALGO_EPISODIC = 7

ONE_HEAD_ABLATION = 2
MUZEROSS = 4
VANILLA = 5
EPISODIC_SIAM_ABLATION = 6

MUZERO_WITH_RND = 8
VK_ABLATION = 9
BOTH = 10
VNOV_ABLATION = 12

#### ENUMS FOR ENVIRONMENTS
FL_4_EASY = -2
FL_5_EASY = -1
FL_6_EASY = 1
FL_6_MEDIUM = 2
FL_6_HARD = 3
FL_8_EASY = 4
FL_8_MEDIUM = 5
FL_8_HARD = 6
FL_KEY = 7
FL_ERRG = 8
CAR_SMALL = 9
CAR_MEDIUM = 10
CAR_HARD = 11
MONTEZUMA = 12

class Config:
    def __init__(self,env_code=False,algo_mode=False):
        ## First, create all the child objects
        self.mcts = child()
        self.model = child()
        self.repr = child()
        self.dynamic = child()
        self.prediction = child()
        self.mcts.expl_noise_maxi = child()
        self.mcts.expl_noise_explorer = child()
        self.siam = child()
        self.training = child()
        self.analysis = child()

        ## Set main features specific to environment and algorithms
        self.preset_environment(env_code)
        self.preset_config(algo_mode)

        ## Sets deque length based on number of stacked observations.
        if self.store_prev_actions:
            self.deque_length = self.timesteps_in_obs * 2 - 1
        else:
            self.deque_length = self.timesteps_in_obs
        
        ############ MODEL SETTINGS
        self.model.res_block_kernel_size = 3
        self.norm_state_vecs = False #I don't norm state vecs, unlike main paper, as it hurts performance of RND.

        # Representation
        self.repr.conv1 = {'channels': self.model.state_channels//2,'kernel_size' : 3, 'stride':2, 'padding':1}
        self.repr.conv2 = {'channels': self.model.state_channels, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        
        # Dynamic
        self.dynamic.conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.dynamic.res_blocks = [self.model.state_channels] 
        self.dynamic.reward_conv_channels = self.model.state_channels//2
        self.dynamic.reward_hidden_dim = 128
        self.dynamic.terminal_conv_channels = self.model.state_channels//2
        self.dynamic.terminal_hidden_dim = 64
        
        #Prediction
        self.prediction.res_block = [self.model.state_channels]
        self.prediction.value_conv_channels = 32
        self.prediction.value_hidden_dim = 128
        
        self.prediction.policy_conv_channels = 32
        self.prediction.policy_hidden_dim = 128
        self.prediction.expV_conv_channels = 32
        self.prediction.expV_hidden_dim = 128
        self.prediction.expV_support = [-10,10,51]

        ###RDN
        self.RND_output_vector = 256
        self.RND_loss = 'cosine' #cosine / MSE
        self.prediction_bias = True #whether or not the RND network contains a bias term in its layers. I find its better if it does although effect is negligible.
        
        #### mcts functions
        self.mcts.c1 = 1
        self.mcts.c2 = 19652 
        self.mcts.temperature_init = 1
        self.mcts.temperature_changes ={-1: 1, 3e6: 0.5 } #I effectively don't chnage temperature as i find it unnecessary and potentially harmful.
        self.mcts.manual_over_ride_play_limit = None    #only used in final testing - set to None otherwise
        self.mcts.exponent_node_n = 1 #not used, always set to 1 like paper
        self.mcts.ucb_denom_k = 1 #not used , always set to 1 like paper
        self.mcts.use_policy = True #always 1, just here for analysis purposes when manually playing.
        self.mcts.expl_noise_maxi.dirichlet_alpha = .3 #maxi follows main paper; explorer is slightly higher on exploratory noise in mcts.
        self.mcts.expl_noise_maxi.noise = 0.3
        self.mcts.expl_noise_explorer.dirichlet_alpha = .5
        self.mcts.expl_noise_explorer.noise = 0.5
        self.mcts.model_expV_on_dones = True #whether or not to cancel exploratory values given by terminal does; i find better to do so.
        
        ### general algorithm functions
        #General
        self.update_play_model = 16
        self.gamma = 0.99
        self.calc_n_step_rewards_after_frames = 10000 #when to start using the V^{t+5} as part of the calculation of value for a node. Delaying it ensures stability.
        self.N_steps_reward = 5
        self.start_frame_count = 0
        self.load_in_model = False
        self.analysis.log_metrics = False
        self.device_train = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_selfplay = 'cpu'
        self.detach_expV_calc = True ## I find its better to detach the state before calculating expV; allows for a more stable estimate of extrinsic reward values.
        self.use_new_episode_expV = False #whether to measure novelty in overlapping episodes; if true, a terminal state will have expV value equal to expV average of starting an episode.
        
        ### When to start training v^{nov}; its best not to start straight away as RND value is unstable.
        self.start_training_expV_min = 10000
        self.start_training_expV_max = 20000 #can change.
        self.start_training_expV_siam_override = 0.8 #I start training when simSiam score reaches 0.8 as this is 'stable'.
        self.value_only = False #OBSOLETE, DOESN'T WORK
        
        ### training
        #I adjust self-play works between [min_workers,max_workers] based on the desired frames-trained-to-play ratio.
        self.training.play_workers = 2 #start number of workers self-playing
        self.training.min_workers = 1
        self.training.max_workers = 6
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
        self.training.train_start_batch_multiple = 5
        self.training.prioritised_replay = True
        self.training.ep_to_batch_ratio = [15,16]
        self.training.main_to_rdn_ratio = 2
        self.training.train_to_RS_ratio = 4
        self.training.on_policy_expV = False #Dont use this; attempted to use updated RND values for v^{nov} (calculated RND score during training each time) but this worsens performance.

        #assertions - jsut for checking any errors.
        assert self.exploration_type in ['none', 'episodic','rdn','both', 'vNov_ablation'], 'change exploration type to valid'
        if self.exploration_type == 'none':
            assert self.rdn_beta == [0,0,1] and self.explorer_percentage == 0, 'set rdn_beta to 0 if youre not exploring and explorer percentage to 1'
        assert self.RND_loss in ['cosine','MSE'], 'change RDN loss type'

    def preset_config(self,algo_mode):
        #### This takes in an algo mode; algo mode is specified in the terminal when running the file.
        self.PRESET_CONFIG = algo_mode
        
        if self.PRESET_CONFIG == FULL_ALGO_RND: #FULL_ALGO_RND == EX-MAX-RND.
            self.VK= True
            self.use_two_heads = True
            self.follow_better_policy = 0.5 #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'rdn' #none / instant / full
            self.rdn_beta = [1/6,4/6,4]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True
        
        if self.PRESET_CONFIG in [FULL_ALGO_EPISODIC, EPISODIC_SIAM_ABLATION]: # FULL_ALGO_EPISODIC == EX-MAX-EPI
            self.VK= True
            self.use_two_heads = True
            self.follow_better_policy = 0.5 #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'episodic' #none / instant / full / episodic
            self.rdn_beta = [1/3,2,6]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True
            self.contrast_vector = True if self.PRESET_CONFIG == FULL_ALGO_EPISODIC else False

        if self.PRESET_CONFIG == ONE_HEAD_ABLATION: #explorer and maximiser share prediction and policy heads, just parameterised by RND beta value.
            #specifically, this is an ablation from the RND algorithm
            self.VK= True
            self.use_two_heads = False
            self.follow_better_policy = 0. #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'rdn' #none / instant / full
            self.rdn_beta = [1/6,4/6,4]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True
        
        if self.PRESET_CONFIG == MUZEROSS:
            self.VK= False
            self.follow_better_policy = 0. #can be 0 to inactivate it.
            self.use_two_heads = False
            self.use_siam = True
            self.exploration_type = 'none' #none / instant / full
            self.rdn_beta = [0,.0,1]
            self.explorer_percentage = 0.
            self.reward_exploration = False
            self.train_dones = True
        
        if self.PRESET_CONFIG == VK_ABLATION:
            self.VK= False
            self.use_two_heads = True
            self.follow_better_policy = 0.5 #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'rdn' #none / instant / full
            self.rdn_beta = [1/6,4/6,4]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True

        
        
        if self.PRESET_CONFIG == MUZERO_WITH_RND:
            self.VK_ceiling = False
            self.VK= False
            self.use_two_heads = False
            self.follow_better_policy = 0. #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'rdn' #none / instant / full
            self.rdn_beta = [0.5, 0.5 , 1]
            self.explorer_percentage = .8
            self.reward_exploration = True
            self.train_dones = True

        if self.PRESET_CONFIG == VANILLA:
            self.VK_ceiling = False
            self.VK= False
            self.use_two_heads = False
            self.use_siam = False
            self.exploration_type = 'none' #none / instant / full
            self.rdn_beta = [0,.0,1]
            self.explorer_percentage = 0.
            self.follow_better_policy = 0.
            self.reward_exploration = False
            self.train_dones = False

        if self.PRESET_CONFIG == VNOV_ABLATION:
            self.VK= True
            self.use_two_heads = True
            self.follow_better_policy = 0.5 #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'vNov_ablation' #none / instant / full
            self.rdn_beta = [1/4,1,4]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True
        
        if self.PRESET_CONFIG == BOTH:
            self.VK= True
            self.use_two_heads = True
            self.follow_better_policy = 0.5 #can be 0 to inactivate it.
            self.use_siam = True
            self.exploration_type = 'both' #none / instant / full
            self.rdn_beta = [1/3,2,6]
            self.explorer_percentage = 0.8
            self.reward_exploration = False
            self.train_dones = True 
            self.contrast_vector = True

        if self.PRESET_CONFIG in [MUZERO_WITH_RND, ONE_HEAD_ABLATION, FULL_ALGO_RND, VK_ABLATION, VNOV_ABLATION,BOTH]:
            #USING RND and therefore need the bigger replay buffer for exploration.
            if self.atari_env:
                self.training.replay_buffer_size = 200 * 1000
                self.training.replay_buffer_size_exploration = 400 * 1000
                self.training.all_time_buffer_size = 400 * 1000
            else:
                self.training.replay_buffer_size = 50 * 1000
                self.training.replay_buffer_size_exploration = 200 * 1000
                self.training.all_time_buffer_size = 200 * 1000
        else:
            self.training.replay_buffer_size = 50 * 1000
            self.training.replay_buffer_size_exploration = 55 * 1000
            self.training.all_time_buffer_size = 55 * 1000
    
    def preset_environment(self, env_code):
        #ENV_CODE specified when running file in termainl
        self.rgb_im = True
        self.channels = 3
        self.model.state_channels = 64
        self.state_size = [6, 6]
        self.timesteps_in_obs = 2
        self.store_prev_actions = True
        
        if env_code == FL_4_EASY:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [4,4]
            self.observable_size = [4,4]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = easy_version4x4()
                self.max_steps = 30
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 255 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.state_size = [4,4]
            self.memory_size = 40

        if env_code == FL_5_EASY:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [5,5]
            self.observable_size = [5,5]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = easy_version5x5()
                self.max_steps = 40
            self.actions_size = 4
            self.optimal_score = 1
            self.total_frames = 305 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.state_size = [6,6]
            self.memory_size = 40

        if env_code == FL_6_EASY:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [6,6]
            self.observable_size = [6,6]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = easy_version6x6()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 105 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 40

        if env_code == FL_6_EASY:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [6,6]
            self.observable_size = [6,6]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = easy_version6x6()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 105 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 40
            
        if env_code == FL_6_MEDIUM:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [6,6]
            self.observable_size = [6,6]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = medium_version1_6x6()
                self.max_steps = 105
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 105 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 40

        if env_code == FL_6_HARD:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [6,6]
            self.observable_size = [6,6]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = medium_version2_6x6()
                self.max_steps = 105
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 105 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 40
        
        if env_code == FL_8_EASY:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [8,8]
            self.observable_size = [8,8]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = easy_version8x8()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 205 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 60
        
        if env_code == FL_8_MEDIUM:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [8,8]
            self.observable_size = [8,8]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = medium_version1_8x8()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 205 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 60

        if env_code == FL_8_HARD:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [8,8]
            self.observable_size = [8,8]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = hard_version8x8()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 255 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 60

        if env_code == FL_KEY:
            ##### ENVIRONMENT
            self.env = keyWorld
            self.same_env_each_time=True
            self.env_size = [7,7]
            self.observable_size = [7,7]
            self.game_modes = 2
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = key1()
                self.max_steps = 120
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 255 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 40}
            self.atari_env = False
            self.memory_size = 30
        
        if env_code == FL_ERRG:
            ##### ENVIRONMENT
            self.env = gridWorld
            self.same_env_each_time=True
            self.env_size = [5,12]
            self.observable_size = [5,12]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = erroneous_with_second_goal()
                self.max_steps = 100
            self.actions_size = 5
            self.optimal_score = 1
            self.total_frames = 255 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 25}
            self.atari_env = False
            self.memory_size = 60

        if env_code == CAR_SMALL:
            ##### ENVIRONMENT
            self.env = RaceWorld
            self.same_env_each_time=True
            self.env_size = [6,30]
            self.observable_size = [6,9]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = small()
                self.max_steps = 100
            self.actions_size = 7
            self.optimal_score = 0.86
            self.total_frames = 205 * 1000
            self.exp_gamma = 0.95
            self.mcts.sims = {-1:6,6000: 50}
            self.atari_env = False
            self.memory_size = 30

        if env_code == CAR_MEDIUM:
            ##### ENVIRONMENT
            self.env = RaceWorld
            self.same_env_each_time=True
            self.env_size = [7,42]
            self.observable_size = [7,9]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = medium()
                self.max_steps = 150
            self.actions_size = 7
            self.optimal_score = 0.86
            self.total_frames = 205 * 1000
            self.exp_gamma = 0.975
            self.mcts.sims = {-1:6,6000: 50}
            self.atari_env = False
            self.memory_size = 30

        if env_code == CAR_HARD:
            ##### ENVIRONMENT
            self.env = RaceWorld
            self.same_env_each_time=True
            self.env_size = [8,60]
            self.observable_size = [8,10]
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = large()
                self.max_steps = 200
            self.actions_size = 7
            self.optimal_score = 0.86
            self.total_frames = 305 * 1000
            self.exp_gamma = 0.975 
            self.mcts.sims = {-1:6,6000: 50}
            self.atari_env = False
            self.memory_size = 30

        if env_code == MONTEZUMA:
            self.env = "MontezumaRevengeNoFrameskip-v4"
            self.same_env_each_time=True
            self.env_size = 'NA'
            self.observable_size = 'NA'
            self.game_modes = 1
            if self.same_env_each_time:
                #### TO BE EDITED FOR EACH MAP.
                self.env_map = 'NA'
                self.max_steps = 'NA'
            self.actions_size = 8
            self.optimal_score = 0.86
            self.total_frames = 45 * 1000
            self.exp_gamma = 0.975 
            self.mcts.sims = {-1:6,6000: 50}
            self.atari_env = True
            self.memory_size = 150

        

        if self.atari_env:
            self.reward_clipping = True
            self.dynamic.reward_support = [-1,1,3]
            self.prediction.value_support = [-10,10,101]
            self.memory_size = 300
            self.image_size = [96,96]
            self.siam.proj_l1 = 1024
            self.siam.proj_out = 1024
            self.siam.pred_hid = 512
            self.repr.res_block_channels = [32, 32,64, 64,64]
            self.repr.res_block_ds = [False, True, False,False, False]
            self.eval_x_frames = 25000
            self.eval_count = 5
            self.training.batch_size = 256
            self.analysis.log_states = False
        
        else:
            self.reward_clipping = False
            self.dynamic.reward_support = [-1,1,51]
            self.prediction.value_support = [-1,1,51]
            self.image_size = [48,48]
            self.siam.proj_l1 = 256
            self.siam.proj_out = 256
            self.siam.pred_hid = 128
            self.repr.res_block_channels = [32, 32,64, 64,64]
            self.repr.res_block_ds = [False, True, False,False, False]
            self.eval_x_frames = 10000
            self.eval_count = 20
            self.training.batch_size = 256
            self.analysis.log_states = True
            self.exploration_logger_dims = (self.game_modes,self.env_size[0]*self.env_size[1])