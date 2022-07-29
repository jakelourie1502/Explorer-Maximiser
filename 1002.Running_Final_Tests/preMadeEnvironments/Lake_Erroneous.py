import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function

def erroneous():
    return np.array([
                        ['S', 'F', 'H', 'H', 'H', 'H', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'H', 'F', 'F','F','F','F','G'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'H', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'F', 'H', 'H', 'H', 'H', 'H','H','H','H','H']
    ]
    )                        

def erroneous_with_second_goal():
    return np.array([
                        ['S', 'F', 'H', 'H', 'H', 'H', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'H', 'F', 'F','F','F','F','G'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'H','H','H','H','H'],
                        ['F', 'F', 'H', 'H', 'H', 'H', 'H', 'H','H','H','H','H'],
                        ['F', 'F', 'E', 'H', 'H', 'H', 'H', 'H','H','H','H','H']
    ]
    )   