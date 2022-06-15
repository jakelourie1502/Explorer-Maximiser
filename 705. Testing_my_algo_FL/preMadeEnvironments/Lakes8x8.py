import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map as env_gen_function

def easy_version8x8():
    return np.array([
                        ['S', 'H', 'F', 'F', 'F', 'H', 'F', 'F'],
                        ['F', 'F', 'F', 'H', 'F', 'H', 'H', 'F'],
                        ['H', 'F', 'F', 'H', 'F', 'F', 'F', 'H'],
                        ['H', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],
                        ['F', 'F', 'F', 'H', 'H', 'F', 'H', 'F'],
                        ['H', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'H', 'F', 'F', 'G']])
def medium_version1_8x8():
    return np.array([['S', 'F', 'F', 'F', 'H', 'F', 'H', 'F'],
                        ['F', 'H', 'H', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'H', 'F'],
                        ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'H', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'H', 'F', 'F', 'F', 'F'],
                        ['F', 'H', 'F', 'F', 'F', 'F', 'H', 'H'],
                        ['H', 'H', 'H', 'F', 'F', 'F', 'F', 'G']])
def medium_version2_8x8():
    return np.array([
                        ['S', 'H', 'F', 'F', 'F', 'F', 'H', 'F'],
                        ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'F', 'F', 'F'],
                        ['F', 'F', 'H', 'F', 'H', 'H', 'F', 'F'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
                        ['H', 'H', 'H', 'F', 'F', 'F', 'H', 'H'],
                        ['F', 'F', 'F', 'F', 'H', 'H', 'H', 'F'],
                        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'G']])
def hard_version8x8():
    return np.array([
                    ['S', 'F', 'H', 'F' ,'F', 'F', 'F', 'F'],
                    ['F', 'H', 'F', 'F', 'F', 'F', 'F', 'H'],
                    ['F', 'H', 'F' ,'F' ,'H' ,'F', 'F' ,'H'],
                    ['F', 'F', 'F' ,'F' ,'H' ,'F', 'H' ,'H'],
                    ['F', 'F', 'F' ,'H' ,'H' ,'F', 'F' ,'F'],
                    ['F', 'F', 'H' ,'F' ,'F' ,'H', 'F' ,'F'],
                    ['F', 'F', 'H' ,'F' ,'F' ,'H', 'H' ,'F'],
                    ['F', 'F', 'F' ,'F' ,'F' ,'H', 'H' ,'G']
                    ]
                    ) 