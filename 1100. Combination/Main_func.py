from threading import Thread
import time
import json
import numpy as np
import os
np.set_printoptions(suppress=True, precision=3)
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(".")
from game_play.play_episode import Episode
from models.ExpMax import ExpMax
from training_and_data.replay_buffer import Replay_Buffer
from training_and_data.training import RDN_loss
from training_and_data.Compute_Batch_Loss import compute_BL
from config import  Config, child
from training_flags import Training_flags
import torch 
torch.set_printoptions(sci_mode=False, precision=4)   
from utils import global_expl, Scores
np.set_printoptions(suppress=True)
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default='1')
parser.add_argument('--env', type=int, default=0)
parser.add_argument('--algo', type=int, default=0)
args = parser.parse_args()

test_run=str(args.name) #sets output file
sys.stdout = open(f"unit_tests/output_{test_run}.txt","w") #sets unit test output file.

class Ep_counter:
    #this is to give unique ids to each episode#
    def __init__(self):
        self.ids = []

class Main:
    def __init__(self):
        self.cfg = Config(args.env, args.algo)
        self.printing_config(self.cfg)
        if not self.cfg.atari_env: print(self.cfg.env_map)
        self._initialise_models() 
        self._init_episode_params()
        self._init_training_params()
        self.training_flags = Training_flags(self, self.cfg)
        self._init_buffer_rdn_RS()
        if self.cfg.analysis.log_states: self.explorer_log = global_expl(self.cfg)
        self.ep_history = []
        self.q_tracker = child()
        self.q_tracker.end_states = 0
        self.q_tracker.non_end_states = 0
        self.start_time = time.time()
        self.curr_best_score = 0
        
        self.reached_optimal_score = False

    def _initialise_models(self):
        """
        Initialise a train model and a model for self-play
        The self-play model will update with the weights from the trainable model every X=16 frames
        Create optimizer
        """
        cfg = self.cfg
        self.ExpMaxTrain = ExpMax(cfg.device_train,self.cfg).to(cfg.device_train)
        self.ExpMaxPlay = ExpMax(cfg.device_selfplay,self.cfg).to(cfg.device_selfplay)
        self.ExpMaxPlay.load_state_dict(self.ExpMaxTrain.state_dict())
        if self.cfg.load_in_model:
            print('loaded in model from saved file')
            self.ExpMaxTrain = torch.load(f'saved_models/jake_zero{test_run}')
            self.ExpMaxPlay = torch.load(f'saved_models/jake_zero{test_run}')
        else:
            with open(f'TestResults/results_{test_run}.txt', 'w') as f:
                print("EVALUATIONS AND MAIN RESULTS", file=f)
        self.ExpMaxTrain.train()
        self.ExpMaxPlay.eval()
        self.optimizer = self.cfg.training.optimizer(self.ExpMaxTrain.parameters(), lr=self.cfg.training.lr, momentum = self.cfg.training.momentum, weight_decay = self.cfg.training.l2 )
        self.compute_batch_loss = compute_BL
        

    def _init_buffer_rdn_RS(self):
        """
        Initialise replay buffer and random network distillation object for tracking running average"""
        self.replay_buffer = Replay_Buffer(self.cfg, self.training_flags)
        self.ep_counter = Ep_counter()
        self.rdn_obj = RDN_loss(self.cfg)
        self.scores = Scores(self.cfg) #keeps track of scores for explorer and maximiser
        
    
    def _init_episode_params(self):
        """
        Some basic params and counts
        """
        self.training_started = False
        self.pick_best = False
        self.tn = time.time()
        self.frame_count = self.cfg.start_frame_count
        self.training_step_counter = self.frame_count * 0.117 #used for loading in the model from a saved point.
        self.resampling_step_counter = self.training_step_counter // self.cfg.training.train_to_RS_ratio
        self.fired_actor_timestamp = time.time()
        self.evaluations = self.frame_count // 10000
        self.e = self.frame_count // 50
        self.test_scores = {}

    def _init_training_params(self):
        """Loading in coefficients and training parameters for ease from config"""
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
        self.thresholdhitter, self.threshold_misser = 0,0

    def actor_wrapper(self):
        """Actor wrapper allows creation of threads that continously self-play episodes"""
        while self.frame_count < self.cfg.total_frames:
            if self.training_flags.self_play_flag: #This checks if we need to pause to wait for training thread
                self.Play_Episode_Wrapper(eval=False)
            else:
                if self.thread_count > self.cfg.training.min_workers and\
                time.time() - self.fired_actor_timestamp > 60: #deletes threads if we have to wait too long for training.
                    self.fired_actor_timestamp = time.time()
                    self.thread_count -=1
                    print(f"deleting a thread, now have {self.thread_count} threads")
                    print("Frames: ", self.frame_count, "train batches done: ",self.training_step_counter, "episodes: ", self.e)
                    sys.stdout.flush()
                    break
                time.sleep(15)
                
        del self            
    

    def Play_Episode_Wrapper(self, eval): #this is the method called by the actor
        """Eval: Boolean
        Set eval to true for evaluation runs
        Set eval to false for normal self-play
        This method plays episodes, and then loads data into the replay buffer (termed 'metrics' here)
        """
        with torch.no_grad():
            
            if eval:
                with torch.no_grad():
                    ep = Episode(self.ExpMaxPlay,self.cfg, scores = self.scores,ep_counter=self.ep_counter, epoch=self.frame_count,rdn_obj = self.rdn_obj, test_mode=True, q_tracker =self.q_tracker, current_best_score = self.curr_best_score)
                    metrics, rew, ep_explr_log, first_move_Qe= ep.play_episode()
                    self.evaluation_reward.append(rew)
                    self.replay_buffer.add_ep_log(metrics)
            else:
                with torch.no_grad():
                    ep = Episode(self.ExpMaxPlay,self.cfg, scores = self.scores,ep_counter=self.ep_counter, epoch=self.frame_count,rdn_obj = self.rdn_obj, test_mode=False,q_tracker =self.q_tracker,current_best_score = self.curr_best_score)
                    metrics, rew,ep_explr_log, first_move_Qe = ep.play_episode()
                    self.rdn_obj.new_ep_expV_deki.append(first_move_Qe)
                    self.replay_buffer.add_ep_log(metrics)
                    if self.cfg.analysis.log_states: 
                        self.explorer_log.append(ep_explr_log)
                    self.ep_history.append(rew)
                    self.frame_count += len(metrics['v'])
                    self.e +=1 
        del self
        
            
    def run(self):
        """Main Function
        Initialises and runs the self-play, evaluation and Training threads"""
        self.losses = []
        self.actor_threads = []

        flag_thread = Thread(target = self.training_flags.run, args=()) #This is a thread that keeps track of which of the main threads should be running.
        flag_thread.start()

        for _ in range(self.cfg.training.play_workers):
            actor_thread = Thread(target=self.actor_wrapper, args=())
            self.actor_threads.append(actor_thread)
            actor_thread.start()
        self.thread_count = self.cfg.training.play_workers

        for _ in range(1):
            trainer_thread = Thread(target=self.compute_batch_loss, args=(self,))
            trainer_thread.start()
        
        for _ in range(1):
            evaluation_thread = Thread(target = self.run_evaluation, args=())
            evaluation_thread.start()
        
    
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
                    if not self.cfg.atari_env:
                        for _ in range(self.cfg.eval_count):
                            thread = Thread(target=self.Play_Episode_Wrapper, args=(True,))
                            t.append(thread)
                            thread.start()
                        for thread in t:
                            thread.join()
                            del thread
                    else:
                        for _ in range(self.cfg.eval_count):
                            self.Play_Episode_Wrapper(True)
                        
                with open(f'TestResults/results_{test_run}.txt', 'a') as f:
                    print("Test reward: ", np.mean(self.evaluation_reward), file=f)
                    self.test_scores[self.cfg.eval_x_frames * (self.frame_count//self.cfg.eval_x_frames)] = np.mean(self.evaluation_reward)
                    with open(f'TestScores/{test_run}.json', 'a') as fp:
                        json.dump(self.test_scores, fp)
                    self.scores.last_test_scores = np.mean(self.evaluation_reward)
                    print("Q value #end_state_>_threshold: ", self.q_tracker.end_states, file=f)
                    print("Q value #non_end_state_>threshold: ", self.q_tracker.non_end_states, file=f)
                    print('maxi score: ', self.scores.scores['maxi'].ma, file=f)
                    print("siam score: ", np.mean(self.siam_log),file=f)
                    self.curr_best_score = max(self.scores.last_test_scores, self.scores.scores['maxi'].ma)
                    print("Scores: ", {str(b): self.scores.scores[str(b)].ma for b in np.round(np.linspace(*self.cfg.rdn_beta),3)}, file=f)
                    print("best rdn ma / adv: ", self.scores.best_rdn_ma, self.scores.best_rdn_adv, file=f)
                    print("best rdn adv: ", self.scores.best_adv, file=f)
                    if self.cfg.env == 'MontezumaRevengeNoFrameskip-v4':
                        print("maxi_rooms: ", self.scores.scores['maxi'].ma, file=f)
                        print("explorer rooms: ", {str(b): self.scores.scores[str(b)].rooms_ma for b in np.round(np.linspace(*self.cfg.rdn_beta),3)}, file=f)
                    print("rdn probs: ", self.scores.probs,file=f)
                    print(f'Episodes: {self.e}, frames: {self.frame_count}, time: {time.time()-self.tn}', file=f)
                    print("training steps: ", self.training_step_counter, file=f)
                    print("retraining steps: ", self.resampling_step_counter, file=f)
                    print(f'RDN obj mus: {[x.mu for x in self.rdn_obj.deki_stats.values()]}', file=f)
                    print(f'RDN obj sigmas: {[x.sigma for x in self.rdn_obj.deki_stats.values()]}', file=f)
                    print(self.e, ":", np.mean(self.ep_history[-self.cfg.training.play_workers*10:]), file=f)
                    print("LR: ", self.lr, file=f)
                    print("replay buffer size: ", len(self.replay_buffer.done_logs), file=f)
                    print('Time Taken : ', (time.time() - self.start_time) // 60, " mins", (time.time() - self.start_time) % 60, " seconds", file=f)

                    if self.cfg.analysis.log_states: 
                        print(self.explorer_log.log.reshape(self.cfg.game_modes, self.cfg.env_size[0],self.cfg.env_size[1]), file=f)
                sys.stdout.flush()
                self.RDN_loss_log = []
                
                #This section enables early stopping if optimal policy is found
                if self.reached_optimal_score:
                    pass
                else:
                    if np.mean(self.evaluation_reward) == self.cfg.optimal_score:
                        self.reached_optimal_score=True
                        self.cfg.total_frames = self.frame_count + 4.5 * self.cfg.eval_x_frames
                        #  = int(max(self.frame_count, self.cfg.total_frames - 4.5*self.cfg.eval_x_frames))
            else:
                time.sleep(1*20 )

    def printing_config(self, cfg):
        #method for printing config objects recursively.
        for a, b in vars(cfg).items():
            if type(b) != child:
                print(f'{a}:{b}')
            else:
                self.printing_config(b)
if __name__ == '__main__':
    Algo = Main()
    Algo.run()