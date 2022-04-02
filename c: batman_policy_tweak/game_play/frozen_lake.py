import numpy as np
from IPython.display import clear_output
import sys
sys.path.append("..")
from global_settings import randomly_create_env, env_size, lakes, goals, star, deque_length, dist, max_plays
from collections import deque

class EnvironmentModel:
    
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state)

        return next_state, reward

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed = None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps
        
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1./n_states) #returns even distribution
    
class gridWorld(Environment):
    
    def __init__(self, size=env_size,lakes=lakes,goals=goals, n_actions = 5, max_steps = max_plays, dist = dist, seed = None, rnd=0, lake_cov=[0.15, 0.25], randomly_create_env = False,star=star):
        self.deki = deque([],deque_length)
        self.size = size
        self.h = size[0]
        self.w = size[1]
        self.corner_idxs = [0,self.w-1,(self.h-1)*self.w,self.h*self.w-1]
        self.random_create = randomly_create_env
        n_states = (size[0]*size[1])+1
        Environment.__init__(self, n_states, n_actions, max_steps,dist)
        self.action_dict = {"Up":0, "Right":1, "Down":2,"Left":3,"DO_NOTHING":4}
        self.chance = rnd
        self.rnd = rnd
        self.has_star = star
        if self.random_create:
            self.pick_random_goal()
            self.set_starting_pos()
            if star:
                self.create_star()
            else:
                self.star = []
            self.lakes = []
            self.create_dicts_and_indexes() #repeat with new lakes
            self.lake_cov = np.random.uniform(lake_cov[0],lake_cov[1])
            self.generate_random_lakes(self.lake_cov)    
        
        else:            
            self.state = np.random.choice(list(range(self.n_states)),p=self.dist)
            self.goal_states = goals
            
            self.lakes = lakes
            self.create_dicts_and_indexes()
            self.goal_idx = self.coors_to_stateIdx[list(self.goal_states.keys())[0]]
            if self.has_star:
                self.create_star()
            else: self.star = []
        self.create_board()
        self._init_probs_dict()
        self.n_steps = 0
    
    def pick_random_goal(self):
        tl, tl_idx = (0,0) , 0
        tr, tr_idx = (0,self.w-1) , self.corner_idxs[1]
        bl, bl_idx = (self.h-1,0) , self.corner_idxs[2]
        br, br_idx = (self.h-1,self.w-1) , self.corner_idxs[3]
        options = [tl, tr, bl, br]
        indexes = [tl_idx, tr_idx, bl_idx, br_idx]
        array = np.array([0,1,2,3])
        chosen = np.random.choice(array)
        self.goal_states = {options[chosen]:1}
        self.goal_idx = indexes[chosen]

    def set_starting_pos(self):
        if self.goal_idx == self.corner_idxs[0]:
            self.state = self.corner_idxs[3]
        if self.goal_idx == self.corner_idxs[3]:
            self.state = self.corner_idxs[0]
        if self.goal_idx == self.corner_idxs[1]:
            self.state = self.corner_idxs[2]
        if self.goal_idx == self.corner_idxs[2]:
            self.state = self.corner_idxs[1]

    def create_star(self):
        star_dist = list(np.linspace(0,self.n_states-2, self.n_states-1))
        star_dist.remove(self.state)
        star_dist.remove(self.goal_idx)
        star_dist = np.array(star_dist)
        star_probs = np.ones_like(star_dist)
        star_probs /= np.sum(star_probs)
        self.star = []
        self.star.append(np.random.choice(star_dist, p=star_probs))
        
    def reset(self):
        obs = self.board.copy()
        blank_board = np.zeros((self.h, self.w))
        posR, posC = self.stateIdx_to_coors[self.state]
        if self.state != self.terminal_state:
            blank_board[posR, posC] = 1.0
        obs = np.stack((obs, blank_board),axis=0)
        for _ in range(deque_length):
            self.deki.append(obs)
        O = np.array(self.deki).reshape(-1,obs.shape[-2],obs.shape[-1])
        return O

    def step(self, action):
        try:
            if action < 0 or action >= self.n_actions:
                raise Exception('Invalid_action.')
        except:
            print("Here",action)
        self.n_steps += 1
        self.state, reward = self.draw(self.state, action)
        done = (self.n_steps >= self.max_steps) or (self.state == self.terminal_state)
        obs = self.board.copy()
        player_board = np.zeros((self.h,self.w))
        posR, posC = self.stateIdx_to_coors[self.state]
        if self.state != self.terminal_state:
            player_board[posR, posC] = 1.0
        obs = np.stack((obs, player_board),0)
        self.deki.append(obs)
        O = np.array(self.deki).reshape(-1,obs.shape[-2],obs.shape[-1])
        return O, self.state, reward, done    


    def p(self,next_state, state, action):
        """
        Here, based on a 'chosen' action, we give the probability of transitioning from one state to another
        Functions:
          We calculate the probability if the chosen action is the 'actual' action, multiplied by chance of not taking random action
          We then add the probability for each action, multiplied by chance of taking random action / number of actions
        """
        no_rnd = 1 - self.rnd
        probas = 0
        probas += no_rnd * self.SAS_probs[state][action][next_state]
        for a in range(self.n_actions):
            probas += (self.rnd/self.n_actions) * self.SAS_probs[state][a][next_state]
        return probas
        "The method p returns the probability of transitioning from state to next state given action. "
        
    def r(self, next_state, state):
        "The method r returns the expected reward in having transitioned from state to next state given action."
        if state == self.goal_idx:
            return 1
        elif state in self.lakes_idx:
            return 0
        elif state in self.star:
            self.star = []
            self.create_board()
            return 0.5
        else:
            return 0.0
    
    def render(self):
        board = self.board.copy()
        player_board = np.zeros((self.h,self.w))
        posR, posC = self.stateIdx_to_coors[self.state]
        if self.state != self.terminal_state:
            player_board[posR, posC] = 1.0
        board = np.stack((board, player_board),0)
        print(board)
        
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
        self.lakes_idx = [self.coors_to_stateIdx[x] for x in self.lakes]
        self.goal_states_idx = {self.coors_to_stateIdx[k]:v for k,v in self.goal_states.items()}


    def create_board(self):
        """
        Inputs: size of lake (h and w), coordinate location of lakes, and coordinate location and value of goal states
        Outputs: array of player-less board, with lake locations and reward locations
        """
        ### Creation of board object
        h,w = self.h,self.w
        self.board = np.array([0.0] * h*w).reshape(h,w)
        for l in self.lakes:
            self.board[l] = -1.0
        for g, r in self.goal_states.items():
            self.board[g] = 1.0
        for s in self.star:
            self.board[self.stateIdx_to_coors[int(s)]] = 0.5
    
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
        lakes_and_goals = list(self.goal_states_idx.keys()) + self.lakes_idx
        
        for state in range(self.n_states):
            SA_prob_dict[state] = {}
            #### Set the chance of entering an absorbing from lake or goal to 1
            for i in range(self.n_actions):
                SA_prob_dict[state][i] = np.zeros((self.n_states,))
                if state in lakes_and_goals or state == self.terminal_state:
                    SA_prob_dict[state][i][self.terminal_state] = 1
            
            if state not in lakes_and_goals and state != self.terminal_state:
                SA_prob_dict[state][self.action_dict['DO_NOTHING']][state] = 1
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
        
    def generate_random_lakes(self,p):
        self.lakes = []
        number_of_lakes = int((self.n_states-1)*p) 
        possible_locations = list(range(self.n_states-1))
        possible_locations.remove(self.state)
        if self.has_star: possible_locations.remove(self.star[0])
        for s in self.goal_states_idx.keys():
            possible_locations.remove(s)
            
        for i in range(number_of_lakes):
            l = np.random.choice(possible_locations)
            self.lakes.append(self.stateIdx_to_coors[l])
            possible_locations.remove(l)
        self.create_dicts_and_indexes()        
        if not self.board_is_playable():
            self.generate_random_lakes(p)
            
    def board_is_playable(self):
        if self.goal_states_idx == 0 or self.state == 0:
            if 1 in self.lakes_idx and self.w in self.lakes_idx:
                return False
        if self.goal_states_idx == self.w-1 or self.state == self.w-1:
            if self.w-2 in self.lakes_idx and 2*self.w-1 in self.lakes_idx:
                return False
        if self.goal_states_idx == self.w*(self.h-1) or self.state == self.w*(self.h-1):
            if self.w*(self.h-1)+1 in self.lakes_idx and self.w*(self.h-2) in self.lakes_idx:
                return False
        if self.goal_idx == self.w*self.h - 1 or self.state == self.w*self.h - 1:
            if self.w*self.h - 2 in self.lakes_idx and self.w*self.h-1 - self.w in self.lakes_idx:
                return False
        return True    


from global_settings import randomly_create_env, env_size, lakes, goals, actions_size, max_plays, dist, play_random, lake_coverage
if __name__ == '__main__':
    
    env=gridWorld(env_size,lakes,goals, n_actions = actions_size, max_steps = max_plays, 
                        dist = dist, seed = None, rnd=play_random, lake_cov=lake_coverage,randomly_create_env= randomly_create_env)
    print(env.goal_idx)
    print(env.lakes_idx)
    done = False
    obs = env.reset()
    print(obs)
    while not done:
        
        act = int(input("give me an action"))
        obs, state, reward, done = env.step(act)
        print(reward)
        print(obs)
        

