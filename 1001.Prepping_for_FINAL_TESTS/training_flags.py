import numpy as np
import torch
import time
from config import child

class Training_flags:
    def __init__(self, algo, cfg):
        self.algo = algo
        self.cfg = cfg
        self.self_play_flag = True
        self.started_training = False
        self.train_flag = False
        self.started_resampling = False
        self.resampling_flag = False
        self.train_head_start_over_resampling = 0
        self.head_start_set = False
        self.add_more_self_play_workers = False
        self.expV_train_flag = False
        self.expV_training_start_flag = self.cfg.total_frames

    def run(self):
        while self.algo.frame_count < self.cfg.total_frames:
            self.set_self_play()
            self.set_resample()
            self.set_train()
            self.set_expV_start_flag()
            if np.random.uniform(0,25) < 1:
                print(f"Training Flag: {self.train_flag}")
                print(f"Self play flag: {self.self_play_flag}")
                print(f"resampling flag: {self.resampling_flag}")
                print("add more workers flag: ", self.add_more_self_play_workers)
                print("expV_train_flag: ", self.expV_train_flag)
                print("expV_train_start_flag: ", self.expV_training_start_flag)
            time.sleep(15)

    def set_expV_start_flag(self):
        """here we set the start point for expV
        we only want to set it again if the frame count is smaller than the starting flag because otherwise it means we set it already
        then we set it if either a) frame count is above expV min and we've hit the overload or b) frame count is above expV max
        """
        if (self.algo.frame_count < self.expV_training_start_flag) and \
            ((self.algo.frame_count > self.algo.cfg.start_training_expV_min and  -np.mean(self.algo.siam_log) > self.algo.cfg.start_training_expV_siam_override) or\
            self.algo.frame_count > self.algo.cfg.start_training_expV_max):
        
            self.expV_training_start_flag = self.algo.frame_count
            print("STARTED EXPV TRAINING ON FRAME NO. ", self.algo.frame_count)
        """Then, if the frame count is 16*batch size more than the start_flag_point, set expV_train_flag to True"""
        if self.algo.frame_count > self.expV_training_start_flag + 16 * self.cfg.training.batch_size:
            self.expV_train_flag = True

    def set_self_play(self):
        frames_trained_to_frames_played_ratio = (self.algo.training_step_counter*self.algo.cfg.training.batch_size) / (self.algo.frame_count+1)
        min_train_to_frame_ratio = self.algo.cfg.training.ep_to_batch_ratio[0]
        
        if self.started_training and frames_trained_to_frames_played_ratio < min_train_to_frame_ratio:
            self.self_play_flag = False
        else:
            self.self_play_flag = True
    
    def set_resample(self):
        if self.algo.cfg.training.resampling and self.algo.frame_count > self.algo.cfg.training.rs_start:
            self.started_resampling = True
            print("Set resampling start")
            if not self.head_start_set:

                self.train_head_start_over_resampling = self.algo.training_step_counter
                self.head_start_set = True
                print(f"Set head start: {self.train_head_start_over_resampling}")
        
        if self.cfg.training.resampling and self.started_resampling and\
            self.algo.training_step_counter - self.train_head_start_over_resampling + 5 > self.algo.resampling_step_counter * self.algo.cfg.training.train_to_RS_ratio:
            self.resampling_flag = True
        else:
            self.resampling_flag = False

    def set_train(self):
        min_frames_to_start_training = self.algo.cfg.training.batch_size*self.algo.cfg.training.train_start_batch_multiple
        frames_trained_to_played_ratio = (self.algo.training_step_counter*self.algo.cfg.training.batch_size) / (self.algo.frame_count+1)
        max_train_to_play_ratio = self.algo.cfg.training.ep_to_batch_ratio[1]
        if self.algo.frame_count > min_frames_to_start_training:
            self.started_training = True
        if self.started_training and\
            frames_trained_to_played_ratio < max_train_to_play_ratio:
            self.add_more_self_play_workers = False

            if self.cfg.training.resampling and self.started_resampling:
                if self.algo.training_step_counter - self.train_head_start_over_resampling <= self.algo.resampling_step_counter * 1.25 * self.algo.cfg.training.train_to_RS_ratio + 5:
                    self.train_flag = True
                else:
                    self.train_flag = False
            else:
                self.train_flag = True
        else:
            self.train_flag = False
            self.add_more_self_play_workers = True