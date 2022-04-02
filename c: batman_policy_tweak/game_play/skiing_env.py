from ast import In
from random import random
import numpy as np
import sys
sys.path.append("..")
from global_settings import env_size, off_ramp, start_state, time_penalty,horizon_view,pre_made_world, deque_length, actions_size, max_plays
import matplotlib.pyplot as plt
from collections import deque
import time
# from global_settings import randomly_create_env, env_size, lakes, goals, star
    
class skiWorld():
    
    def __init__(self, env_size =env_size, off_ramp = off_ramp, n_actions = actions_size, max_steps = max_plays, dist = None, seed = None,view_game=False, generate = False,start_state=start_state):
        self.deki = deque([],deque_length)
        self.pre_made_env =  pre_made_world
        self.generate = generate
        self.length= env_size[1]
        self.width = env_size[0]
        self.n_states = env_size[0]*env_size[1] + 1
        self.state_vector = np.zeros((self.n_states+1)) #includes terminal
        self.n_actions = n_actions
        self.action_dict = {0:"Up", 1:"Accelerate", 2:"Down",3:"Brake",4: "Hard_Up", 5: "Hard_Down", 6:"Do_Nothing"}
        self.off_ramp = off_ramp
        self.state = start_state
        self.create_dicts()
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.generate_cliffs_and_holes()
        self.generate_goal_states()
        self.velocity = [0,0] ### down, right
        self.cum_reward = 0
        self.create_board()
        self.view_game = view_game
        self.n_steps = 0
        self.max_steps = max_steps
        

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
    
    def generate_cliffs_and_holes(self):
        if self.generate:
            self.generate_cliffs()
            self.generate_holes()
        else:
            self.cliffs_idx = []
            self.holes_idx = []
            for idx, i in enumerate(self.pre_made_env.flatten()):
                if i == 0:
                    self.cliffs_idx.append(idx)
                elif i == .7:
                    self.holes_idx.append(idx)
            self.cliffs_coors = [self.stateIdx_to_coors[x] for x in self.cliffs_idx]
            self.holes_coors = [self.stateIdx_to_coors[x] for x in self.holes_idx]
            
        
    def generate_cliffs(self):
        
        #3 and 4 are used to stop the first section becoming cliffs
        self.cliffs_idx = [x for x in range(3,self.length-17)] + [x for x in range(self.length*(self.width-1)+3,self.n_states-17)]
        
        for i in range(3):
            temp_list = []
            for c in self.cliffs_idx:
                if np.random.uniform(0,1) > 0.66:
                    if c < self.n_states - self.length -1:
                        temp_list.append(c+self.length)
                    if c > self.length:
                        temp_list.append(c-self.length)
            self.cliffs_idx.extend(temp_list)
        self.cliffs_idx = list(set(self.cliffs_idx))
        self.cliffs_coors = [self.stateIdx_to_coors[x] for x in self.cliffs_idx]
        
        
    
    def generate_holes(self):
        
        self.holes_idx = []
        for slot in range(self.n_states-1):
            y_coor = self.stateIdx_to_coors[slot][1]
            if  y_coor > 3 and y_coor < self.length - 17: ##stops the first section becoming holey
                if np.random.uniform(0,1) > 0.9:
                    self.holes_idx.append(slot)
        self.holes_coors = [self.stateIdx_to_coors[x] for x in self.holes_idx]

    
    def generate_goal_states(self):
        self.goal_states = [x for x in range(self.n_states-1) if self.stateIdx_to_coors[x][1] > (self.length-17) and self.stateIdx_to_coors[x][0] not in [0,7]]
        

    def create_board(self):
        ### Creation of board object
        w,l = self.width,self.length
        self.board = np.array([1.0] * l*w).reshape(w,l)
        for c in self.cliffs_coors:
            self.board[c[0],c[1]] = 0
        for h in self.holes_coors:
            self.board[h[0],h[1]] = 0.7
        for g in self.goal_states:
            c = self.stateIdx_to_coors[g]
            self.board[c[0],c[1]] = -1
        self.board[self.stateIdx_to_coors[self.state]] = 0.3
        
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
        elif self.state in self.goal_states:
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
                reward = -3*time_penalty
            else:
                self.n_steps +=1
                reward = -time_penalty
        self.cum_reward += reward

        done = (self.n_steps >= self.max_steps) or (self.state == self.terminal_state)
        
        if not done:
            obs = self.create_board()
            view_idx = self.stateIdx_to_coors[self.state][1]
            obs = obs[:,view_idx-1:view_idx+horizon_view-1]
            self.deki.append(obs)
        else:
            self.deki.append(self.deki[-1]) #sticks last one on again. it will get deleted anyway tbf.
        
        if self.view_game and not  done:
            time.sleep(0.1)
            print(self.stateIdx_to_coors[self.state])
            O = np.array(self.deki).reshape(-1,env_size[0],horizon_view)
            # print(O)
            plt.close()
            # plt.imshow(obs,'gray')
            # plt.show(block=False)
            sys.stdout.flush()
        

        
        # try:
        #     print('here',done, np.array(self.deki).shape, self.n_steps)
        # except:
        #     print(done, self.n_steps, 'printing issue', len(self.deki), self.deki[0].shape)
        O = np.array(self.deki).reshape(-1,env_size[0],horizon_view)
        return O, self.state, reward, done    

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
        
    def reset(self):
        obs = self.create_board()
        
        view_idx = self.stateIdx_to_coors[self.state][1]
        
        obs = obs[:,view_idx-1:view_idx+horizon_view-1]
        
        if self.view_game:
            print(self.stateIdx_to_coors[self.state])
            # print(obs)
            # plt.imshow(obs,'gray')
            # time.sleep(5)
            # plt.show(block=False)
        
        for _ in range(deque_length):
            self.deki.append(obs)
        O = np.array(self.deki).reshape(-1,obs.shape[-2],obs.shape[-1])
        return O
    
    
    
if __name__ == '__main__':
    
    env=skiWorld(view_game=True)
    done = False
    obs = env.reset()
    # print(env.goal_states)

    while not done:
        act = int(input("Give it to me"))
        obs, state, reward, done = env.step(act)
        print(env.n_steps)
        # plt.imshow(obs)
        
        print(reward)
        print(env.cum_reward)
        
        

