import numpy as np
import torch
import time
from config import child

class Training_flags:
    """
    This is a class dedicated to keeping the ratio between frames-trained-to-played in check with our target
    It also deletes threads where necessary.
    It also controls when we start training v^{nov} (labelled expV).
     """
    def __init__(self, algo, cfg):
        self.algo = algo
        self.cfg = cfg
        self.self_play_flag = True #should we be doing selfplay
        self.started_training = False #have we started training
        self.train_flag = False #should we be training right now.
        self.head_start_set = False #obsolete
        self.add_more_self_play_workers = False #to add more threads
        self.expV_train_flag = False #whether we should be training v^{nov}
        self.expV_training_start_flag = self.cfg.total_frames #placeholder - will change when we want to start training v^{nov}

    def run(self):
        ### Runs continuously every 15 seconds.
        while self.algo.frame_count < self.cfg.total_frames:
            self.set_self_play()
            self.set_train()
            self.set_expV_start_flag()
            if np.random.uniform(0,25) < 1:
                print(f"Training Flag: {self.train_flag}")
                print(f"Self play flag: {self.self_play_flag}")
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
            ((self.algo.frame_count > self.algo.cfg.start_training_expV_min+self.cfg.start_frame_count and  -np.mean(self.algo.siam_log) > self.algo.cfg.start_training_expV_siam_override) or\
            self.algo.frame_count > self.algo.cfg.start_training_expV_max+self.cfg.start_frame_count):
        
            self.expV_training_start_flag = self.algo.frame_count
            print("STARTED EXPV TRAINING ON FRAME NO. ", self.algo.frame_count)
        """Then, if the frame count is 16*batch size more than the start_flag_point, set expV_train_flag to True"""
        if self.algo.frame_count > self.expV_training_start_flag + 16 * self.cfg.training.batch_size:
            self.expV_train_flag = True

    def set_self_play(self):
        ### We want to be doing selfplay if the ratio is within an acceptable range.
        frames_trained_to_frames_played_ratio = (self.algo.training_step_counter*self.algo.cfg.training.batch_size) / (self.algo.frame_count+1)
        min_train_to_frame_ratio = self.algo.cfg.training.ep_to_batch_ratio[0]
        
        if self.started_training and frames_trained_to_frames_played_ratio < min_train_to_frame_ratio:
            self.self_play_flag = False
        else:
            self.self_play_flag = True
    

    def set_train(self):
        ## we want to be training after 'min frames to start training' and the ratio is within an acceptable range.
        min_frames_to_start_training = self.algo.cfg.training.batch_size*self.algo.cfg.training.train_start_batch_multiple + self.cfg.start_frame_count
        frames_trained_to_played_ratio = (self.algo.training_step_counter*self.algo.cfg.training.batch_size) / (self.algo.frame_count+1)
        max_train_to_play_ratio = self.algo.cfg.training.ep_to_batch_ratio[1]
        if self.algo.frame_count > min_frames_to_start_training:
            self.started_training = True
        if self.started_training and\
            frames_trained_to_played_ratio < max_train_to_play_ratio:
            self.add_more_self_play_workers = False


            self.train_flag = True
        else:
            self.train_flag = False
            self.add_more_self_play_workers = True