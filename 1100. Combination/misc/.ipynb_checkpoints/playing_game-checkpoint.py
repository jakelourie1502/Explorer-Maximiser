# from msvcrt import kbhit
import time
from pickle import FALSE
import sys
sys.path.append('..')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from math import factorial as F
import matplotlib.pyplot as plt
from utils import support_to_scalar
import numpy as np
import torch
from training_and_data.training import  loss_func_v, loss_func_p, loss_func_r, loss_func_proj, RDN_loss,loss_func_future_nov, Normaliser
from utils import global_expl, get_epoch_params, Scores
# from training_and_data.training import loss_func_RDN
sys.stdout = open("playing_game.txt","w")

from config import Config
cfg = Config()
from utils import vector2d_to_tensor_for_model
from game_play.play_episode import Episode
siam_scores = np.zeros((cfg.training.k)) + 0.5
norman = Normaliser()
scores = Scores()
class Ep_counter:
    def __init__(self):
        self.ids = []
ep_c = Ep_counter()
rdn_ob = RDN_loss()
ExpMax = torch.load('../saved_models/jake_zero',map_location=torch.device('cpu'))
def Play_Episode_Wrapper(eval):
    with torch.no_grad():
        ep = Episode(ExpMax,0,cfg,siam_score= siam_scores, norman = norman, actor_scores = scores,ep_counter=ep_c, epoch=50000)
        if eval:
            with torch.no_grad():
                ep.actor = 'maxi'; ep.actor_id = 0
                metrics, rew, ep_explr_log= ep.play_episode(rdn_ob, False, True)
                print(metrics)

for _ in range(1):
    Play_Episode_Wrapper(True)