import numpy as np
from IPython.display import clear_output
import sys

from torch import ge
sys.path.append("..")

from collections import deque
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym import Env, spaces, utils
from gym.utils import seeding
from .frozen_lake_imagery import FL_Image
### 

class gridWorld(Env):
    
    def __init__(self,cfg):
        self.images = FL_Image(cfg)
        self.cfg = cfg
        self.side_length = [self.cfg.env_size[0],self.cfg.env_size[1]]
        self.slippage = 0.
        self.f_perc = 0.6
        self.max_steps = self.cfg.max_steps
        env_size = [self.side_length[0], self.side_length[1]]
        self.env_size = env_size
        self.observable_size = self.env_size
        self.n_actions = self.cfg.actions_size
        self.goal_locked = True

        ### GYM FEATURES
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.side_length[0], self.side_length[1], 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.n_actions)
        self.reward_range = [0,1]
        ####

        self.size = env_size
        self.h = self.size[0]
        self.w = self.size[1]
        self.corner_idxs = [0,self.w-1,(self.h-1)*self.w,self.h*self.w-1]
        
        self.n_states = env_size[0]*env_size[1]+1
        self.action_dict = {"Up":0, "Right":1, "Down":2,"Left":3,"PICK_UP_KEY":4}
        self.create_dicts_and_indexes()
        self.random_state = np.random.RandomState(42)
        if self.cfg.same_env_each_time:
            self.gym_view = self.cfg.env_map
            
        else:
            #obsolete with this view
            self.gym_view = np.array([y for x in generate_random_map(self.side_length,self.f_perc) for y in x]).reshape(self.side_length,self.side_length)
        
        self.log_locations()
        self.starting_state = self.state
        self.create_board()
        self._init_probs_dict()
        self.n_steps = 0
    
    def log_locations(self):
        self.lakes = []
        self.key = -1
        for idx, i in enumerate(self.gym_view.flatten()):
            
            if i == 'H':
                self.lakes.append(idx)
            elif i == 'S':
                self.state = idx
            elif i == 'G':
                self.goal_state = idx
            elif i == 'K':
                self.key = idx

        self.lakes_coors = [self.stateIdx_to_coors[x] for x in self.lakes]
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.goal_coors = self.stateIdx_to_coors[self.goal_state]
        self.key_coors = self.stateIdx_to_coors[self.key]


    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state)

        return next_state, reward

        
    def reset(self , *, seed = None,  return_info: bool = False, options = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.state = self.starting_state
        self.log_locations()
        
        self.state_coors = self.stateIdx_to_coors[self.state]
        self.create_board()
        self.n_steps = 0
        self.goal_locked = True
        self._init_probs_dict()
        obs = self.board.copy()
        
        return obs.transpose(2,0,1)


    def step(self, action):
        try:
            if action < 0 or action >= self.n_actions:
                raise Exception('Invalid_action.')
        except:
            print("Here",action)
        self.n_steps += 1
        self.state, reward = self.draw(self.state, action)
        if self.state == self.key and action == self.action_dict['PICK_UP_KEY']:
            self.key = -1
            self.key_coors = [-1, -1]
            self.goal_locked = False
            self._init_probs_dict()

        self.state_coors = self.stateIdx_to_coors[self.state]
        
        done = (self.n_steps >= self.max_steps) or (self.state == self.terminal_state)
        
        self.create_board()
        o = self.board.copy()
        return o.transpose(2,0,1), reward, done, {"stnum": [1-self.goal_locked,self.state]}


    def p(self,next_state, state, action):
        """
        Here, based on a 'chosen' action, we give the probability of transitioning from one state to another
        Functions:
          We calculate the probability if the chosen action is the 'actual' action, multiplied by chance of not taking random action
          We then add the probability for each action, multiplied by chance of taking random action / number of actions
        """
        no_rnd = 1 - self.slippage
        probas = 0
        probas += no_rnd * self.SAS_probs[state][action][next_state]
        for a in range(self.n_actions):
            probas += (self.slippage/self.n_actions) * self.SAS_probs[state][a][next_state]
        return probas
        "The method p returns the probability of transitioning from state to next state given action. "
        
    def r(self, next_state, state):
        "The method r returns the expected reward in having transitioned from state to next state given action."
        
        if state in [self.goal_state] and not self.goal_locked:
            return 1
        elif state in self.lakes:
            return 0
        else:
            return 0.0
    
    def render(self):
        print(self.board)
    
    def close(self):
        pass

    def seed(self, seed = None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def create_dicts_and_indexes(self):
        """
        Inputs... 
         size of lake (tuple e.g. (4,4))
         Location of lakes in coordinate form e.g. [(0,1),(1,2)...]
         Location of goal_states and their rewards e.g. {(3:3):1, (5,5):-1} In our examples this is always just one goal state

        Outputs...
         Dictionary linking coordinates to index of each state, and reverse dictionary
         Lake squares in index form e.g. [3,6,9]
         Goal states in index form e.g {15: 1, 25: -1}
        """
        
        self.coors_to_stateIdx = {}
        idx =0
        for r in range(self.h):
            for c in range(self.w):
                self.coors_to_stateIdx[(r,c)] = idx

                idx+=1

        self.coors_to_stateIdx[(-1,-1)] = self.n_states-1
        self.terminal_state = self.n_states-1

        self.stateIdx_to_coors = {}
        for k,v in self.coors_to_stateIdx.items():
            self.stateIdx_to_coors[v]=k
        


    def create_board(self):
        """
        Inputs: size of lake (h and w), coordinate location of lakes, and coordinate location and value of goal states
        Outputs: array of player-less board, with lake locations and reward locations
        """
        ### Creation of board object
        self.board = self.images.create_board_image(self.lakes, [self.goal_state], self.state, [self.key])
    

    def _init_probs_dict(self):
        """
        In: the backend of the board (stateIdx_to_coors dict, lakes, goals, terminal state)
        Out: returns the impact of an ACTUAL action on the board position of a player
        Structure of output: {Current_State1: {Up: state1, state2, state 3....,
                            Down: state1, state2, state 3...}
                            ....
                    Current_State2: {Up ......}}
        
        note: 'actual' action distinguished here from 'chosen' action. Players 'choose', then we apply randomness, and then there is an 'actual' action
        This function concerns the effect of an 'actual' action on the position of a player.
        """
        
        ### HELPER FUNCTIONS
        def state_is_top(state):
            return self.stateIdx_to_coors[state][0] == 0
        def state_is_bottom(state):
            return self.stateIdx_to_coors[state][0] == self.h-1
        def state_is_left(state):
            return self.stateIdx_to_coors[state][1] == 0
        def state_is_right(state):
            return self.stateIdx_to_coors[state][1] == self.w-1
        def move_up():
            return -self.w
        def move_down():
            return self.w
        def move_left():
            return -1
        def move_right():
            return 1
        
        SA_prob_dict = {}
        
        for state in range(self.n_states):
            SA_prob_dict[state] = {}
            #### Set the chance of entering an absorbing from lake or goal to 1
            for i in range(self.n_actions):
                SA_prob_dict[state][i] = np.zeros((self.n_states,))
                if state in self.lakes or (state == self.goal_state and not self.goal_locked) or state == self.terminal_state:
                    SA_prob_dict[state][i][self.terminal_state] = 1
            
            if state not in self.lakes and (state != self.goal_state or self.goal_locked) and state != self.terminal_state:
                SA_prob_dict[state][self.action_dict['PICK_UP_KEY']][state] = 1
                """For UP"""
                if not state_is_top(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Up']][state+move_up()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Up']][state] = 1

                """For DOWN"""
                if not state_is_bottom(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Down']][state+move_down()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Down']][state] = 1

                """For LEFT"""
                if not state_is_left(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Left']][state+move_left()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Left']][state] = 1

                """For RIGHT"""
                if not state_is_right(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Right']][state+move_right()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Right']][state] = 1     
        self.SAS_probs = SA_prob_dict
    

if __name__ == '__main__':
    from config import Config as cfg
    import matplotlib.pyplot as plt
    c = cfg()
    
    env=gridWorld(c)
    for _ in range(2):
        
        done = False
        obs = env.reset()
        plt.imshow(obs.transpose(1,2,0))
        plt.show(block=False)
        
        while not done:
            
            act = int(input("give me an action"))
            obs, reward, done, info = env.step(act)
            plt.close()
            plt.imshow(obs.transpose(1,2,0))
            plt.show(block=False)
            print(reward)
            print(done)
        

