import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function


def key1():
    #7, 7
    return np.array([
                        ['H', 'H', 'H', 'H', 'H','F','G'],
                        ['F', 'F', 'H', 'H', 'F','F','F'],
                        ['F', 'F', 'F', 'F', 'F','F','F'],
                        ['S', 'F', 'H', 'H', 'H','H','H'],
                        ['F', 'F', 'H', 'H', 'F','F','F'],
                        ['F', 'F', 'F', 'F', 'F','K','F'],
                        ['F', 'F', 'H', 'H', 'F','F','F']])

