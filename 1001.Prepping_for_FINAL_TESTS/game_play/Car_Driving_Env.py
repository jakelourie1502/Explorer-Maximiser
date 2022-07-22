from ast import In
from random import random
import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from collections import deque
import time
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym import Env, spaces, utils
from gym.utils import seeding
# from global_settings import randomly_create_env, env_size, lakes, goals, star
from .Car_Driving_imagery import Car_Image

class RaceWorld(Env):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.images = Car_Image(cfg)
        self.env_size = self.cfg.env_size
        self.observable_size = self.cfg.observable_size
        self.start_state = self.env_size[1]*3+1
        self.max_plays = self.cfg.max_steps
        self.time_penalty = 1/self.max_plays
        self.pre_made_env =  self.cfg.env_map
        self.length= self.env_size[1]
        self.width = self.env_size[0]
        self.n_states = self.env_size[0]*self.env_size[1] + 1
        # self.state_vector = np.zeros((self.n_states+1)) #includes terminal
        self.n_actions = self.cfg.actions_size
        self.action_dict = {0:"Up", 1:"Accelerate", 2:"Down",3:"Brake",4: "Hard_Up", 5: "Hard_Down", 6:"Do_Nothing"}
        self.state = self.start_state
        self.create_dicts()
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.generate_cliffs_and_holes_and_goals()
        self.velocity = [0,0] ### down, right
        self.cum_reward = 0
        self.create_board()
        self.n_steps = 0
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.observable_size[0], self.observable_size[1], 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.n_actions)
        self.reward_range = [-1,1]
        

    def create_dicts(self): 
        self.coors_to_stateIdx = {}
        idx =0
        for r in range(self.width):
            for c in range(self.length):
                self.coors_to_stateIdx[(r,c)] = idx
                idx+=1

        self.coors_to_stateIdx[(-1,-1)] = self.n_states-1
        self.terminal_state = self.n_states-1

        self.stateIdx_to_coors = {}
        for k,v in self.coors_to_stateIdx.items():
            self.stateIdx_to_coors[v]=k
    
    def generate_cliffs_and_holes_and_goals(self):
        self.cliffs_idx = []
        self.holes_idx = []
        self.goals_idx = []
        for idx, i in enumerate(self.pre_made_env.flatten()):
            if i == 0:
                self.cliffs_idx.append(idx)
            elif i == 2:
                self.holes_idx.append(idx)
            elif i == 3:
                self.goals_idx.append(idx)
        self.cliffs_coors = [self.stateIdx_to_coors[x] for x in self.cliffs_idx]
        self.holes_coors = [self.stateIdx_to_coors[x] for x in self.holes_idx]
        self.goals_coors = [self.stateIdx_to_coors[x] for x in self.goals_idx]


    def create_board(self):
        ### Creation of board object
        ##### OK so this creates the board as we see it.
        self.board = self.images.create_board_image(self)

        return self.board

    def step(self, action):
        action=int(action)
        try:
            if action < 0 or action >= self.n_actions:
                raise Exception('Invalid_action.')
        except:
            print("Here",action)
        
        self.calculate_velocity(action)
        
        if self.state in self.cliffs_idx: 
            self.state = self.terminal_state
            reward = -1 - self.cum_reward
        elif self.state in self.goals_idx:
            self.state = self.terminal_state
            reward = 1
        else:
            if self.velocity != [0,0]:
                if self.velocity[1] == 0 or self.velocity[0] == 0: #going in a straight line
                    
                    speed = max(abs(self.velocity[1]),abs(self.velocity[0]))
                    individual_steps = [self.velocity[0]//speed,self.velocity[1]//speed]
                    for _ in range(speed):
                        self.state_coors = (self.state_coors[0] + individual_steps[0], self.state_coors[1] + individual_steps[1])
                        self.state = self.coors_to_stateIdx[self.state_coors]
                        if self.state in self.holes_idx or self.state in self.cliffs_idx:
                            break
                else:
                    
                    self.state_coors = (self.state_coors[0] + self.velocity[0], self.state_coors[1] + self.velocity[1])
                    self.state = self.coors_to_stateIdx[self.state_coors]
            if self.state in self.holes_idx:
                self.n_steps += 3
                reward = -3*self.time_penalty
            else:
                self.n_steps +=1
                reward = -self.time_penalty
        

        done = (self.n_steps >= self.max_plays) or (self.state == self.terminal_state)
        if done and self.state != self.terminal_state:
            reward = -1 - self.cum_reward 
        
        self.cum_reward += reward
        
        if not done:
            obs = self.create_board()
            self.last_obs = obs
        else:
            obs = self.last_obs #this logic is just to not get it to fuck up at the end
        
        return obs.transpose(2,0,1), reward, done, {'stnum': [0,self.state]}

    def calculate_velocity(self,action):
        
        if self.state in self.holes_idx:
            
            self.velocity = [0,0]
        vel = self.velocity
        fwd_speed = vel[1]
        side_speed = vel[0] #down is positive
        
        if fwd_speed == 0: #if you've done a last extra turn, set speed to 0 globally.
            side_speed = 0

        if fwd_speed >= 1:
            if self.action_dict[action] == "Up":
                if side_speed == 1:
                    side_speed = 0
                    fwd_speed = 1
                elif side_speed == -1:
                    side_speed = -1
                    fwd_speed = 0
                elif side_speed == 0:
                    fwd_speed = 1
                    side_speed = -1
            if self.action_dict[action] == 'Hard_Up':
                if side_speed == 1:
                    fwd_speed = 1
                    side_speed = -1
                if side_speed == -1:
                    side_speed = -1
                    fwd_speed = 0
                if side_speed == 0:
                    fwd_speed = 0 
                    side_speed = -1
            if self.action_dict[action] == "Down":
                if side_speed == 1:
                    side_speed = 1
                    fwd_speed = 0
                elif side_speed == -1:
                    side_speed = 0
                    fwd_speed = 1
                elif side_speed == 0:
                    fwd_speed = 1
                    side_speed = 1
            if self.action_dict[action] == 'Hard_Down':
                if side_speed == 1:
                    fwd_speed = 0
                    side_speed = 1
                if side_speed == -1:
                    side_speed = 1
                    fwd_speed = 1
                if side_speed == 0:
                    fwd_speed = 0 
                    side_speed = 1        
        if self.action_dict[action] == "Brake":
                
                fwd_speed = fwd_speed // 2
                if side_speed != 0:
                    side_sign = side_speed / np.abs(side_speed)
                    side_speed =  side_sign * (np.abs(side_speed) // 2)
        if self.action_dict[action] == 'Accelerate':
            
            fwd_speed = min(2, fwd_speed+1)

            side_speed = 0
        
        if self.state in self.cliffs_idx:
            self.velocity = [0,0]
        else:
            self.velocity = [side_speed, fwd_speed]
        

    
        
    def reset(self , *, seed = None,  return_info: bool = False, options = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.state = self.start_state
        self.state_coors = self.stateIdx_to_coors[self.start_state]
        self.n_steps = 0
        obs = self.create_board()
        # view_idx = self.stateIdx_to_coors[self.state][1]
        # obs = obs[:,view_idx:view_idx+self.observable_size[1]]
        return obs.transpose(2,0,1)
    
    def render(self):
        print(self.board)
    
    def close(self):
        pass
    
    def seed(self, seed = None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    from config import Config
    env=RaceWorld(Config())
    done = False
    obs = env.reset()
    # print(env.goal_states)
    print(obs.shape)
    while not done:
        plt.close()
        plt.imshow(obs.transpose(1,2,0))
        plt.show(block=False)
        act = int(input("Give it to me"))
        obs, reward, done, info = env.step(act)
        print(env.n_steps)
        # plt.imshow(obs)
        
        print(reward)
        print(env.cum_reward)
        
        

