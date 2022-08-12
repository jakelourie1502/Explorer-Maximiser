from numpy import save, load
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000, precision=2)
a = load('../saved_models_and_rewards/reward_log.npy')
b = len(a)
r = []
span = 200
for s in range(b//span):
    r.append(np.sum(a[s*span:(s+1)*span]))
print(r)