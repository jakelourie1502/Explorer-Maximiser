import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function


def easy_version6x6():
    return np.array([
                        ['S', 'F', 'H', 'H', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'F'],
                        ['F', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'F', 'H'],
                        ['F', 'F', 'F', 'F', 'H', 'H'],
                        ['F', 'F', 'H', 'F', 'F', 'G']])
def medium_version1_6x6():
    return np.array([
                        ['S', 'F', 'H', 'H', 'F', 'F'],
                        ['F', 'F', 'F', 'F', 'H', 'F'],
                        ['F', 'F', 'H', 'H', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'F', 'F'],
                        ['F', 'F', 'F', 'F', 'H', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'G']])
def medium_version2_6x6():
    return np.array([
                        ['S', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'F'],
                        ['F', 'H', 'H', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'H', 'F'],
                        ['F', 'H', 'F', 'F', 'H', 'F'],
                        ['F', 'H', 'F', 'F', 'H', 'G']])
def hard_version6x6():
    return np.array([['S', 'F', 'H', 'H', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'F'],
                        ['F', 'F', 'H', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'H', 'H', 'F'],
                        ['F', 'F', 'F', 'F', 'H', 'G']])