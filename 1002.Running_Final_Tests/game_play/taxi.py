from ast import In
from random import random
import numpy as np
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from collections import deque
import time
# from global_settings import randomly_create_env, env_size, lakes, goals, star

env_size = [8,8]
observable_size = [8,8]


max_plays = 40
deque_length = 1
actions_size = 6

pre_made_world1 = np.array(
            [[0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 3., 0., 4., 0., 0., 0., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [0., 2., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 0., 1., 0., 1.]])

pre_made_world2 = np.array(
            [[0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 2., 1., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [0., 4., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 3., 1., 0., 1.],
            [0., 1., 0., 1., 0., 1., 0., 1.]])

pre_made_world3 = np.array(
            [[0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 4., 1., 0., 1.],
            [0., 0., 0., 0., 0., 2., 0., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 0., 3., 0., 0.],
            [0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 0., 1., 0., 1.]])


class taxiWorld():
    
    def __init__(self, env_size =env_size, n_actions = actions_size, max_steps = max_plays, generate = False):
        
        self.generate = generate
        self.base_reward = 0
        self.drop_off_reward = 1
        self.env_size = env_size
        self.observable_size = observable_size
        self.state = -1
        
        self.pre_made_env1 =  pre_made_world1
        self.pre_made_env2 =  pre_made_world2
        self.pre_made_env3 =  pre_made_world3
        
        self.length= env_size[1]
        self.width = env_size[0]
        self.n_states = env_size[0]*env_size[1] + 1
        self.n_actions = n_actions
        self.action_dict = {0:"Left", 1:"Right", 2:"Up",3:"Down",4: "PickUp", 5: "DropOff"}
        self.create_dicts()
        self.max_steps = max_plays
        
        self.walls = []
        self.non_walls = []
        self.reset()
        

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
    
    def reset(self):
        
        self.cum_reward = 0
        self.board = self.pre_made_env1
        self.create_backend()
        self.n_steps = 0
        self.done = False
        self.picked_up = False
        self.successful_dropoffs = 0
        self.milestones = 0
        return self.create_board(), self.state, 0, self.done

    def distance(self,point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1]-point2[1])

    def create_backend(self):
        flat_board = self.board.reshape(-1)
        if self.generate:

            for idx, i in enumerate(flat_board):
                if i == 1.: 
                    self.walls.append(idx)
                else:
                    self.non_walls.append(idx)
            self.walls_coors = [self.stateIdx_to_coors[x] for x in self.walls]
            self.non_walls_coors = [self.stateIdx_to_coors[x] for x in self.non_walls]
        
            if self.state == -1:
                self.state = np.random.choice(self.non_walls)
                self.state_coors = self.stateIdx_to_coors[self.state]
            self.pickup_coors = self.state_coors
            self.dropoff_coors = self.state_coors
            while self.distance(self.state_coors, self.pickup_coors) < 3:
                self.pickup_point = np.random.choice(self.non_walls)
                self.pickup_coors = self.stateIdx_to_coors[self.pickup_point]
            while self.distance(self.state_coors, self.dropoff_coors) < 3 or self.distance(self.pickup_coors, self.dropoff_coors) < 3:
                self.dropoff_point = np.random.choice(self.non_walls)
                self.dropoff_coors = self.stateIdx_to_coors[self.dropoff_point]
            
        else:
            for idx, i in enumerate(flat_board):
                if i == 1.: 
                    self.walls.append(idx)

                else:
                    self.non_walls.append(idx)
                if i == 2:
                    self.dropoff_point = idx
                    self.dropoff_coors = self.stateIdx_to_coors[self.dropoff_point]
                elif i == 3:
                    self.pickup_point = idx
                    self.pickup_coors = self.stateIdx_to_coors[self.pickup_point]
                elif i ==4:
                    self.state = idx
                    self.state_coors = self.stateIdx_to_coors[self.state]
            self.walls_coors = [self.stateIdx_to_coors[x] for x in self.walls]
            self.non_walls_coors = [self.stateIdx_to_coors[x] for x in self.non_walls]
        self.pickup_point, self.pickup_coors = [self.pickup_point], [self.pickup_coors]
        self.picked_up = False

    def create_next_scene(self):
        if self.generate:
            self.create_backend()
        else:
            if self.successful_dropoffs % 3 == 1:
                self.board = self.pre_made_env2
            elif self.successful_dropoffs % 3 == 2:
                self.board = self.pre_made_env3
            else:
                self.board = self.pre_made_env1
            self.create_backend()

    def create_board(self):
        ### Creation of board object
        board = np.zeros((self.env_size[0]*self.env_size[1]))
        for i in range(len(board)):
            if i in self.walls:
                board[i] = 1
            elif i == self.state:
                board[i] = 4
            elif i == self.dropoff_point:
                board[i] = 2
            elif i in self.pickup_point:
                board[i] = 3
        board = board.reshape(self.env_size[0], self.env_size[1])
        return board
    

    def step(self, action):
        self.action_dict = {0:"Left", 1:"Right", 2:"Up",3:"Down",4: "PickUp", 5: "DropOff"}
        action=int(action)
        
        if action < 0 or action >= self.n_actions:
                raise Exception('Invalid_action.')
        
        if action == 0:
            if (self.state_coors[0], self.state_coors[1]-1) not in self.walls_coors and self.state_coors[1] > 0:
                self.state_coors = (self.state_coors[0], self.state_coors[1]-1)
                self.state = self.coors_to_stateIdx[self.state_coors]
            reward = self.base_reward 
            self.n_steps +=1
        if action == 1:
            if (self.state_coors[0], self.state_coors[1]+1) not in self.walls_coors and self.state_coors[1] < self.width-1:
                self.state_coors = (self.state_coors[0], self.state_coors[1]+1)
                self.state = self.coors_to_stateIdx[self.state_coors]
            
            reward = self.base_reward 
            self.n_steps +=1
            
        if action == 2:
            if (self.state_coors[0]-1, self.state_coors[1]) not in self.walls_coors and self.state_coors[0] > 0:
                self.state_coors = (self.state_coors[0]-1, self.state_coors[1])
                self.state = self.coors_to_stateIdx[self.state_coors]
            reward = self.base_reward 
            self.n_steps +=1
        
        if action == 3:
            if (self.state_coors[0]+1, self.state_coors[1]) not in self.walls_coors and self.state_coors[0] < self.length-1:
                self.state_coors = (self.state_coors[0]+1, self.state_coors[1])
                self.state = self.coors_to_stateIdx[self.state_coors]
            reward = self.base_reward 
            self.n_steps +=1
        
        if action == 4:
            if self.picked_up == False and self.state in self.pickup_point:
                self.milestones += 1
                self.picked_up = True
                
                self.pickup_point = []
                self.pickup_coors = []
            reward = self.base_reward
            self.n_steps +=1
        
        if action == 5:
            if self.picked_up: 
                if self.state == self.dropoff_point:
                    self.milestones += 1
                    self.n_steps  = 10*((self.n_steps // 10) + 1 )
                    self.successful_dropoffs +=1
                    reward = self.drop_off_reward
                    if self.n_steps != self.max_steps:
                        self.create_next_scene()

                else:
                    reward = 0
                    self.n_steps += 1
            else:
                reward = self.base_reward
                self.n_steps +=1
        if self.n_steps >= self.max_steps:
            self.done = True

        o = self.create_board()
            
        return o, self.state, reward, self.done, self.milestones 

    
if __name__ == '__main__':
    
    env=taxiWorld()
    done = False
    obs, _, _, _ = env.reset()
    # print(env.goal_states)

    while not done:
        print(obs)
        act = int(input("Give it to me"))
        obs, state, reward, done = env.step(act)
        print(env.n_steps)
        # plt.imshow(obs)
        
        print(reward)
        print(env.cum_reward)
        
        

