import numpy as np
from IPython.display import clear_output

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
    
    def __init__(self, size,lakes,goals, n_actions = 4, max_steps = 300, dist = None, seed = None, rnd=0.1):
        
        self.size = size
        n_states = (size[0]*size[1])
        Environment.__init__(self, n_states, n_actions, max_steps,dist)
        self.action_dict = {"Up":0, "Right":1, "Down":2,"Left":3}
        self.chance = rnd
        self.rnd = rnd
        self.lakes = lakes
        self.goal_states = goals
        self.h = size[0]
        self.w = size[1]
        self.reset()

    def reset(self):
        self.create_dicts_and_indexes()
        self.create_board()
        self._init_probs_dict()
        self.n_steps = 0
        self.state = np.random.choice(list(range(self.n_states)),p=self.dist)
        obs = self.board.copy()

        posR, posC = self.stateIdx_to_coors[self.state]
        obs[posR, posC] = -1.0
        return obs

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid_action.')
            
        self.n_steps += 1
    
        self.state, reward = self.draw(self.state, action)
        done = (self.n_steps >= self.max_steps) or (self.state in self.lakes_and_goals)
        obs = self.board.copy()
        posR, posC = self.stateIdx_to_coors[self.state]
        obs[posR,posC] = -1
        return obs, self.state, reward, done    


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
        return self.goal_states_idx[next_state] if next_state in self.goal_states_idx else 0
    
    def render(self):
        board = self.board.copy()
        posR, posC = self.stateIdx_to_coors[self.state]
        if self.state != self.terminal_state:
            board[posR, posC] = -1
        clear_output()
        print(board)
        
    def create_dicts_and_indexes(self,terminal_state_exist = False):
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

        if terminal_state_exist:
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
            self.board[l] = 0.5
        for g, r in self.goal_states.items():
            self.board[g] = 1.0
    
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
        self.lakes_and_goals = lakes_and_goals
        
        for state in range(self.n_states):
            SA_prob_dict[state] = {}
            #### Set the chance of entering an absorbing from lake or goal to 1
            for i in range(4):
                SA_prob_dict[state][i] = np.zeros((self.n_states,))
                if state in lakes_and_goals:
                    for act in range(4):
                        SA_prob_dict[state][i][state] = 1
            
            if state not in lakes_and_goals:
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
        number_of_lakes = int((self.n_states)*p) 
        possible_locations = list(range(self.n_states))
        possible_locations.remove(self.state)
        for s in self.goal_states_idx.keys():
            possible_locations.remove(s)
        for i in range(number_of_lakes):
            l = np.random.choice(possible_locations)
            self.lakes.append(self.stateIdx_to_coors[l])
        
        if (list(self.goal_states_idx.keys())[0] - 1 == 0.5 and list(self.goal_states_idx.keys())[0] - self.w == 0.5) or (self.state + 1 == 0.5 and self.state + self.w == 0.5):
            self.generate_random_lakes(self, p)

if __name__ == '__main__':
    size = (12,12)
    goals = {(size[0]-1,size[1]-1):1}
    dist = np.zeros((size[0]*size[1]+1))
    dist[0]=1
    env=gridWorld(size,[],goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
    env.generate_random_lakes(0.2)
    print(env.reset())