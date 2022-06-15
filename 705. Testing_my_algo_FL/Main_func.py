from threading import Thread
import copy
import time

import numpy as np
import os
np.set_printoptions(suppress=True, precision=3)
import sys
from torch.optim import optimizer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from collections import deque
sys.path.append(".")

from game_play.play_episode import Episode
from models.ExpMax import ExpMax
from training_and_data.replay_buffer import Replay_Buffer
from training_and_data.training import  loss_func_v, loss_func_p, loss_func_r, loss_func_proj, RDN_loss,loss_func_expV
from config import  Config, child
import torch 
torch.set_printoptions(sci_mode=False, precision=4)   
from utils import global_expl, get_lr, Scores
np.set_printoptions(suppress=True)
if len(sys.argv) > 1:
    test_run=str(sys.argv[1])
else:
    test_run = '1'
sys.stdout = open(f"TestResults/output_{test_run}.txt","w")

class Ep_counter:
    #this is to give unique ids to each episode#
    def __init__(self):
        self.ids = []

class Main:
    def __init__(self):
        self.cfg = Config()
        print(self.cfg.env_map)
        self._initialise_models() 
        self._init_episode_params()
        self._init_training_params()
        self._init_buffer_rdn_norman()
        if self.cfg.analysis.log_states: 
            self.explorer_log = global_expl()
        self.ep_history = []
        self.q_tracker = child()
        self.q_tracker.end_states = 0
        self.q_tracker.non_end_states = 0
        self.start_time = time.time()

    def _initialise_models(self):
        ### Create 2 models pre model (self_play and train)
        cfg = self.cfg
        self.ExpMaxTrain = ExpMax(cfg.device_train).to(cfg.device_train)
        self.ExpMaxPlay = ExpMax(cfg.device_selfplay).to(cfg.device_selfplay)
        self.ExpMaxPlay.load_state_dict(self.ExpMaxTrain.state_dict())
        if self.cfg.load_in_model:
            print('loaded in model from saved file')
            self.ExpMaxTrain = torch.load(f'saved_models/jake_zero{test_run}')
            self.ExpMaxPlay = torch.load(f'saved_models/jake_zero{test_run}')
        self.ExpMaxTrain.train()
        self.ExpMaxPlay.eval()
        self.optimizer = self.cfg.training.optimizer(self.ExpMaxTrain.parameters(), lr=self.cfg.training.lr, momentum = self.cfg.training.momentum, alpha = self.cfg.training.rho,weight_decay = self.cfg.training.l2)

    def _init_buffer_rdn_norman(self):
        self.replay_buffer = Replay_Buffer(self.cfg)
        self.ep_counter = Ep_counter()
        self.rdn_obj = RDN_loss()
        self.scores = Scores()
    
    def _init_episode_params(self):
        self.training_started = False
        self.pick_best = False
        self.tn = time.time()
        self.frame_count = self.cfg.start_frame_count
        self.training_step_counter = self.frame_count // 10
        self.fired_actor_timestamp = time.time()
        self.evaluations = self.frame_count // 10000
        self.e = self.frame_count // 50

    def _init_training_params(self):
        self.value_coef = self.cfg.training.coef.value
        self.dones_coef = self.cfg.training.coef.dones
        self.siam_coef = self.cfg.training.coef.siam
        self.K = self.cfg.training.k
        self.evaluation_reward= []
        self.support_full_values  = (np.linspace(*self.cfg.prediction.value_support)).reshape(-1,1)
        self.support_full_rewards = (np.linspace(*self.cfg.dynamic.reward_support)).reshape(-1,1)
        self.support_expV = (np.linspace(*self.cfg.prediction.expV_support)).reshape(-1,1)
        self.thread_count = self.cfg.training.play_workers
        self.lr = 0
    
    def actor_wrapper(self):
        while self.frame_count < self.cfg.total_frames:
            
            
            frames_played_to_frames_trained_ratio = (self.training_step_counter*self.cfg.training.batch_size) / (self.frame_count+1)
            min_frame_ratio = self.cfg.training.ep_to_batch_ratio[0]
            min_frames_to_start_training = self.cfg.training.batch_size*self.cfg.training.train_start_batch_multiple
            if self.frame_count > 5 * min_frames_to_start_training and frames_played_to_frames_trained_ratio < min_frame_ratio:
                if self.thread_count> self.cfg.training.min_workers:
                    if time.time() - self.fired_actor_timestamp > 60:
                        self.fired_actor_timestamp = time.time()
                        self.thread_count -=1
                        print(f"deleting a thread, now have {self.thread_count} threads")
                        print("Frames: ", self.frame_count, "train batches done: ",self.training_step_counter, "episodes: ", self.e)
                        sys.stdout.flush()
                        break
                time.sleep(30)
            
            self.Play_Episode_Wrapper(eval=False)
        
        del self            
    

    def Play_Episode_Wrapper(self, eval):
        with torch.no_grad():
            
            if eval:
                with torch.no_grad():
                    ep = Episode(self.ExpMaxPlay,self.cfg, scores = self.scores,ep_counter=self.ep_counter, epoch=self.frame_count,rdn_obj = self.rdn_obj, test_mode=True, q_tracker =self.q_tracker)
                    metrics, rew, ep_explr_log, first_move_Qe= ep.play_episode()
                    self.evaluation_reward.append(rew)
                    self.replay_buffer.add_ep_log(metrics)
            else:
                with torch.no_grad():
                    ep = Episode(self.ExpMaxPlay,self.cfg, scores = self.scores,ep_counter=self.ep_counter, epoch=self.frame_count,rdn_obj = self.rdn_obj, test_mode=False,q_tracker =self.q_tracker)
                    metrics, rew,ep_explr_log, first_move_Qe = ep.play_episode()
                    self.rdn_obj.new_ep_expV_deki.append(first_move_Qe)
                    self.replay_buffer.add_ep_log(metrics)
                    if self.cfg.analysis.log_states: 
                        self.explorer_log.append(ep_explr_log)
                    self.ep_history.append(rew)
                    self.frame_count += len(metrics['v'])
                    self.e +=1
            
            
    def run(self):
        self.losses = []
        self.actor_threads = []
        for _ in range(self.cfg.training.play_workers):
            actor_thread = Thread(target=self.actor_wrapper, args=())
            self.actor_threads.append(actor_thread)
            actor_thread.start()
        self.thread_count = self.cfg.training.play_workers

        for _ in range(1):
            trainer_thread = Thread(target=self.compute_batch_loss, args=())
            trainer_thread.start()
        
        for _ in range(1):
            evaluation_thread = Thread(target = self.run_evaluation, args=())
            evaluation_thread.start()


    def compute_batch_loss(self):
        self.siam_log = deque([], 200)
        while self.frame_count < self.cfg.total_frames:
            frames_played_to_frames_trained_ratio = (self.training_step_counter*self.cfg.training.batch_size) / (self.frame_count+1)
            max_frame_ratio = self.cfg.training.ep_to_batch_ratio[1]
            min_frames_to_start_training = self.cfg.training.batch_size*self.cfg.training.train_start_batch_multiple
            if len(self.replay_buffer.action_log) > min_frames_to_start_training  and frames_played_to_frames_trained_ratio < max_frame_ratio:
                self.lr = get_lr(self.training_step_counter)
                self.training_started = True
                self.training_step_counter +=1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                
                self.replay_buffer.purge() #keep replay buffer at reasonable size.
                
                ### getting target data
                sample_obs, indices, weights = self.replay_buffer.get_sample(prioritised_sampling=self.cfg.training.prioritised_replay)
                sample_obs = sample_obs.float()
                s = self.ExpMaxTrain.representation(sample_obs)
                done_tensor_tmin1 = torch.zeros((len(indices),self.K))
                done_tensor_same_t = torch.zeros((len(indices),self.K))
                weights = torch.tensor(weights).to(self.cfg.device_train).reshape(-1,1)
                loss = 0

                for k in range(self.K):
                    action_index = np.array([self.replay_buffer.action_log[x+k] for x in indices])
                    rdn_vals = np.array([self.replay_buffer.rdn_beta[x+k] for x in indices])
                    
                    p , v , expV = self.ExpMaxTrain.prediction(s,rdn_vals)
                    s, r, d = self.ExpMaxTrain.dynamic(s,action_index)
                    s.register_hook(lambda grad: grad * 0.5)
                    #note: episodic memory function can be done with dones last period because if it's dones this period, we still have an action to predict.

                    ###### CAPTURING DONE MASKS (need two sets, as some variables require the prior k done, and some require this k done)
                    ### SAME period
                    dones_k_same_t = np.array([self.replay_buffer.done_logs[x+k] for x in indices])
                    dones_k_same_t = torch.maximum(torch.tensor(dones_k_same_t), done_tensor_same_t[:, k-1]).to(self.cfg.device_train)
                    done_tensor_same_t[:, k] = dones_k_same_t
                    dones_k_same_t_in_format = dones_k_same_t.reshape(-1,1)
                    
                    ### PRIOR period
                    if k == 0:
                        # dones_k_tmin1 = done_tensor_tmin1[:, 0].to(self.cfg.device_train)
                        dones_k_tmin1 = torch.zeros((self.cfg.training.batch_size))
                    else:
                        dones_k_tmin1 = np.array([self.replay_buffer.done_logs[x+k-1] for x in indices])
                        dones_k_tmin1 = torch.maximum(torch.tensor(dones_k_tmin1), done_tensor_tmin1[:, k-1]).to(self.cfg.device_train)
                        done_tensor_tmin1[:, k] = dones_k_tmin1
                    dones_k_tmin1_in_format = dones_k_tmin1.reshape(-1,1)

                    ## TERMINALITY
                    loss_dones_k = - (torch.mean((1-dones_k_tmin1_in_format)*(dones_k_same_t_in_format * torch.log(d+1e-4) + (1-dones_k_same_t_in_format)*torch.log(1-d+1e-4))))
                    # loss_dones_k = - (torch.mean((1)*(dones_k_same_t_in_format * torch.log(d+1e-4) + (1-dones_k_same_t_in_format)*torch.log(1-d+1e-4))))
                    loss+= self.dones_coef * loss_dones_k

                    #### SIAM SECTION
                    if self.cfg.use_siam:
                        o = torch.tensor(np.array([self.replay_buffer.obs[x+k+1] for x in indices])).float()
                        w_grad_head = self.ExpMaxTrain.project(s)
                        if not self.cfg.siam.negative_examples:
                            with torch.no_grad():
                                reps = self.ExpMaxTrain.representation(o)
                                stopped_proj = self.ExpMaxTrain.project(reps, grad_branch = False)
                            loss_siam = loss_func_proj(stopped_proj, w_grad_head, dones_k_same_t_in_format)

                        if self.cfg.siam.negative_examples:
                            idx = torch.randperm(self.cfg.training.batch_size)
                            shuffled_states = s[idx]
                            w_grad_head_fake = self.ExpMaxTrain.project(shuffled_states)
                            if self.cfg.siam.project_rep:
                                if self.cfg.siam.grad_rep:
                                    reps = self.ExpMaxTrain.representation(o)
                                    stopped_proj = self.ExpMaxTrain.project(reps, grad_branch = True)
                                else:
                                    with torch.no_grad():
                                        reps = self.ExpMaxTrain.representation(o)
                                        stopped_proj = self.ExpMaxTrain.project(reps, grad_branch = True).detach()
                            else:
                                with torch.no_grad():
                                    reps = self.ExpMaxTrain.representation(o)
                                    stopped_proj = self.ExpMaxTrain.project(reps, grad_branch = False)
                                    
                            loss_siam = 0.5*loss_func_proj(stopped_proj, w_grad_head, dones_k_same_t_in_format)
                            
                            loss_siam_neg = -0.5*loss_func_proj(stopped_proj, w_grad_head_fake, dones_k_same_t_in_format)#
                            
                            loss_siam += loss_siam_neg
                        
                        loss += self.siam_coef * loss_siam
                        self.siam_log.append(loss_siam.detach().cpu().numpy())
                    
                    #### POLICY
                    true_policy = torch.tensor(np.array([self.replay_buffer.policy_logs[x+k] for x in indices])).to(self.cfg.device_train).reshape(-1,self.cfg.actions_size)
                    loss_Pk = loss_func_p(p, true_policy, dones_k_tmin1_in_format,weights)
                    loss += loss_Pk
                    
                    
                    #### VALUE AND REWARD (non-curiousity)
                    ## reanalyse values
                    if self.frame_count >= self.cfg.calc_n_step_rewards_after_frames:
                        with torch.no_grad():
                            obs_k_plus_N = torch.tensor(np.array([self.replay_buffer.obs[x+k+self.cfg.N_steps_reward] for x in indices])).float()          
                            _, v_reanalyse,_ = self.ExpMaxPlay.prediction(self.ExpMaxPlay.representation(obs_k_plus_N),rdn_vals)
                            v_reanalyse = torch.tensor(v_reanalyse.detach().numpy() @ self.support_full_values)

                            k_ep_id = np.array([self.replay_buffer.ep_id[x+k] for x in indices])
                            k_plus_N_ep_id = np.array([self.replay_buffer.ep_id[x+k+self.cfg.N_steps_reward] for x in indices])
                            mask = (k_ep_id == k_plus_N_ep_id).reshape(-1,1)
                            boot_values = torch.tensor(v_reanalyse * mask) * (self.cfg.gamma ** self.cfg.N_steps_reward)
                        
                        for idx, bv in zip(indices, boot_values[:, 0].cpu().numpy()):
                            self.replay_buffer.n_step_returns_with_V[idx + k] = np.clip(self.replay_buffer.n_step_returns[idx + k] + bv,self.cfg.prediction.value_support[0],self.cfg.prediction.value_support[1])
                        if np.random.uniform(0,10000) < 1:
                            print("Showing what the boot values, the n step returns and the n step returns with V look like")
                            print(boot_values[:10])
                            for idx in indices[:10]:
                                print(self.replay_buffer.n_step_returns[idx+k], self.replay_buffer.n_step_returns_with_V[idx + k])
                    else:
                        boot_values = 0
                        
                    if self.cfg.value_only:
                        true_values = torch.tensor(np.array([self.replay_buffer.n_step_returns[x] for x in indices])).to(self.cfg.device_train).reshape(-1,1) #here, when we just use value, we don't use the dones in the value calculation.
                    else:
                        true_values = torch.tensor(np.array([self.replay_buffer.n_step_returns_with_V[x+k] for x in indices])).to(self.cfg.device_train).reshape(-1,1) 
                        # true_values =  torch.clip(true_values + boot_values, self.cfg.prediction.value_support[0],self.cfg.prediction.value_support[1])
                        true_rewards = torch.tensor(np.array([self.replay_buffer.reward_logs[x+k] for x in indices])).to(self.cfg.device_train).reshape(-1,1)
                        loss_Rk = loss_func_r(r, true_rewards, dones_k_tmin1_in_format, weights)
                        loss += loss_Rk
                    
                    loss_Vk = loss_func_v(v, true_values, dones_k_tmin1_in_format,weights)
                    loss += loss_Vk * self.value_coef

                if np.random.uniform(0,50) < 1 or self.training_step_counter == 2:
                    print("siam score: ", np.mean(self.siam_log))

                loss.backward()
                self.optimizer.step(); self.optimizer.zero_grad()                        
                

                ### RDN SECTION
                if self.training_step_counter % self.cfg.training.main_to_rdn_ratio == 0 and self.cfg.exploration_type != 'none':
                
                #check whether to start training
                    if (self.frame_count < self.cfg.start_training_expV) and \
                        (self.frame_count > self.cfg.start_training_expV_min) and \
                        (-np.mean(self.siam_log) > self.cfg.start_training_expV_siam_override):
                        
                        self.cfg.start_training_expV = self.frame_count
                        print("STARTED EXPV TRAINING ON FRAME NO. ", self.frame_count)
                    
                    if np.random.uniform(0,100) < 1: print("first move QE: ",self.rdn_obj.new_ep_mu)
                    if self.frame_count > self.cfg.start_training_expV + 16 * self.cfg.training.batch_size:
                        epsam = True
                    else:
                        epsam = False
                    sample_obs, indices, weights = self.replay_buffer.get_sample(prioritised_sampling=False, exploration_sampling=epsam)
                    sample_obs = sample_obs.float()
                    s = self.ExpMaxTrain.representation(sample_obs)
                    done_tensor_tmin1 = torch.zeros((len(indices),self.K))
                    done_tensor_same_t = torch.zeros((len(indices),self.K))
                    weights = torch.tensor(weights).to(self.cfg.device_train).reshape(-1,1)
                    loss = 0

                    ##### EXPV
                    expV_capturer = torch.zeros((self.cfg.training.batch_size, self.K, self.cfg.prediction.expV_support[2]))
                    expR_capturer = torch.zeros((self.cfg.training.batch_size, self.K)).detach()
                    
                    for k in range(self.K):
                        action_index = np.array([self.replay_buffer.action_log[x+k] for x in indices])
                        rdn_vals = np.array([self.replay_buffer.rdn_beta[x+k] for x in indices])
                        
                        if self.cfg.detach_expV_calc:
                            _ , _ , expV = self.ExpMaxTrain.prediction(s.detach(),rdn_vals)
                        else:
                            _ , _ , expV = self.ExpMaxTrain.prediction(s,rdn_vals)
                        
                        expV_capturer[:,k] = expV
                        s, _, _ = self.ExpMaxTrain.dynamic(s,action_index)
                        s.register_hook(lambda grad: grad * 0.5)
                        
                        
                        ###### CAPTURING DONE MASKS (need two sets, as some variables require the prior k done, and some require this k done)
                        
                        ## DONES
                        #same period
                        dones_k_same_t = np.array([self.replay_buffer.done_logs[x+k] for x in indices])
                        dones_k_same_t = torch.maximum(torch.tensor(dones_k_same_t), done_tensor_same_t[:, k-1]).to(self.cfg.device_train)
                        done_tensor_same_t[:, k] = dones_k_same_t
                        dones_k_same_t_in_format = dones_k_same_t.reshape(-1,1)

                        #last period
                        if k == 0:
                            dones_k_tmin1 = done_tensor_tmin1[:, 0].to(self.cfg.device_train)
                        else:
                            dones_k_tmin1 = np.array([self.replay_buffer.done_logs[x+k-1] for x in indices])
                            dones_k_tmin1 = torch.maximum(torch.tensor(dones_k_tmin1), done_tensor_tmin1[:, k-1]).to(self.cfg.device_train)
                            done_tensor_tmin1[:, k] = dones_k_tmin1
                        dones_k_tmin1_in_format = dones_k_tmin1.reshape(-1,1)
                        
                        ##### EXPR
                    
                        rdn_random = self.ExpMaxTrain.RDN(s).detach()
                        rdn_pred = self.ExpMaxTrain.RDN_prediction(s.detach())
                        expR_values, loss_rdn_k = self.rdn_obj.train(rdn_random, rdn_pred, dones_k_tmin1_in_format, weights, k+1)
                        loss+= loss_rdn_k * self.cfg.training.coef.rdn
                        if self.cfg.training.on_policy_expV:
                            expR_capturer[:,k] = expR_values.detach()[:,0]
                        else:
                            expR_capturer[:,k] = torch.tensor(np.array([self.replay_buffer.exp_r[x+k] for x in indices]))
                        
                    
                    ###expV finishing section
                    if self.cfg.exploration_type=='full':
                        
                        if epsam: #so this happens after 5,000 frames, before that we just give a small amount. we also then immediately start bootstrapping
                            true_expV_values = torch.zeros((self.cfg.training.batch_size,self.K))
                            true_expV_values += expR_capturer
                            for k in range(1,self.K):
                                true_expV_values[:,:-k] += expR_capturer[:,k:] * (self.cfg.exp_gamma**k)
                    
                            obs_k_plus_N = torch.tensor(np.array([self.replay_buffer.obs[x+self.cfg.training.k] for x in indices])).float()          
                            _, _,K_plus_1_expV = self.ExpMaxPlay.prediction(self.ExpMaxPlay.representation(obs_k_plus_N),rdn_vals)
                            K_plus_1_expV = torch.tensor(K_plus_1_expV.detach().numpy() @ self.support_expV) #bs, 1
                            
                            K_plus_1_expV = (1-dones_k_same_t_in_format) * K_plus_1_expV 
                            if self.cfg.use_new_episode_expV:
                                K_plus_1_expV += dones_k_same_t_in_format * self.rdn_obj.new_ep_mu

                            K_plus_1_expV = torch.squeeze(K_plus_1_expV.detach())
                            for k in range(self.K):
                                true_expV_values[:,-(k+1)] += K_plus_1_expV*self.cfg.exp_gamma**(k+1) #here we are reversing our way through... the kth element will just be r+V* (one n step)
                            
                            true_expV_values = torch.clip(true_expV_values,*self.cfg.prediction.expV_support[:2])

                        else:
                            true_expV_values = torch.randn((self.cfg.training.batch_size, self.K)) * 0.1
                        
                    for k in range(self.K):
                        loss+= self.cfg.training.coef.expV * loss_func_expV(expV_capturer[:,k],true_expV_values[:,k].reshape(-1,1),done_tensor_tmin1[:,k].reshape(-1,1))
                    
                    self.rdn_obj.update()
        
                #### END OF LOSS ACCUMULATION
                    loss.backward()
                    self.optimizer.step(); self.optimizer.zero_grad()
                
                if self.training_step_counter % self.cfg.update_play_model == 0:
                    self.save_and_load_model()
                
            else:
                
                time.sleep(1*20)
                if len(self.replay_buffer.action_log) > min_frames_to_start_training and self.thread_count < self.cfg.training.max_workers:
                    actor_thread = Thread(target=self.actor_wrapper, args=())
                    actor_thread.start()
                    self.thread_count+=1
                    print(f"Adding thread: now have {self.thread_count} threads")
    
    
    def save_and_load_model(self):
        self.ExpMaxPlay.load_state_dict(self.ExpMaxTrain.state_dict())
        self.ExpMaxPlay.rdn_dekis = self.rdn_obj.deki_stats
        
        torch.save(self.ExpMaxPlay, f'saved_models/jake_zero{test_run}')

    def run_evaluation(self):
        
        while self.frame_count < self.cfg.total_frames:
            if self.evaluations * self.cfg.eval_x_frames < self.frame_count:
                print("Starting evaluation")
                sys.stdout.flush()
                self.evaluations +=1
                with torch.no_grad():
                    self.evaluation_reward = []
                    t = []
                    for _ in range(self.cfg.eval_count):
                        thread = Thread(target=self.Play_Episode_Wrapper, args=(True,))
                        t.append(thread)
                        thread.start()
                    for thread in t:
                        thread.join()
                        del thread
                print("Test reward: ", np.mean(self.evaluation_reward))
                print("Q value #end_state_>_threshold: ", self.q_tracker.end_states)
                print("Q value #non_end_state_>threshold: ", self.q_tracker.non_end_states)
                print("Scores: ", self.scores.scores)
                print(f'Episodes: {self.e}, frames: {self.frame_count}, time: {time.time()-self.tn}')
                print("training steps: ", self.training_step_counter)
                print(f'RDN obj mus: {[x.mu for x in self.rdn_obj.deki_stats.values()]}')
                print(f'RDN obj sigmas: {[x.sigma for x in self.rdn_obj.deki_stats.values()]}')
                print(self.e, ":", np.mean(self.ep_history[-self.cfg.training.play_workers*10:]))
                print("LR: ", self.lr)
                print("replay buffer size: ", len(self.replay_buffer.done_logs))
                print('Time Taken : ', (time.time() - self.start_time) // 60, " mins", (time.time() - self.start_time) % 60, " seconds")
                if self.cfg.analysis.log_states: 
                    print(self.explorer_log.log.reshape(-1, self.cfg.env_size[0],self.cfg.env_size[1]))
                sys.stdout.flush()
                self.RDN_loss_log = []
                #### BREAK CLAUSE
                if np.mean(self.evaluation_reward) == 1:
                    self.frame_count = self.cfg.total_frames
            else:
                time.sleep(1*20 )
if __name__ == '__main__':
    Algo = Main()
    Algo.run()