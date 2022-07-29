import gym
import numpy as np
from copy import deepcopy
env = gym.make('MontezumaRevengeNoFrameskip-v4')

class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            # if t == 3:
            #     self.final_frame = state
            reward += r
            if done:
                break
        
        
        return state, reward, done, info

class MontezumaVisitedRoomEnv(gym.Wrapper):
    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()  # Only stores unique numbers.

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        self.visited_rooms.add(ram[self.room_address])
        if done:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"].update(visited_room=deepcopy(self.visited_rooms))
            self.visited_rooms.clear()
        return state, reward, done, info

    def reset(self):
        return self.env.reset()

def make_atari(env_name):
    if env_name == 'MontezumaRevengeNoFrameskip-v4':
        return MontezumaVisitedRoomEnv(RepeatActionEnv(gym.make(env_name)),3)
    return RepeatActionEnv(gym.make(env_name))