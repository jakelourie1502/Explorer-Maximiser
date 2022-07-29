from numpy import load
a = load('0_epoch_analysis_file.npy')
import numpy as np
print(np.max(a[:,3]))
#7 children ; policy, yes/no action, Qe, Q, r, v, exp_r, exp_v, epoch, epoch, sims, actual action, game_step, state_NUM
