# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:50:22 2020

Author: David O'Callaghan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from deep_sea_treasure import DeepSeaTreasureEnvironment

with open('dst_results.pkl', 'rb') as f:
    stats_dict = pickle.load(f)

data = []
for key in stats_dict:
    Q_values = stats_dict[key][1]
    env = DeepSeaTreasureEnvironment()
    state = env.reset()
    r = [0,0]
    while True:
        action = np.argmax(Q_values[state])
        state, rewards, done = env.step(action)
        r[0] += rewards[0]
        r[1] += rewards[1]
        if done:
            print(f'Finished {key}')
            print(f'{r}\n')
            data.append(r)
            break
        
data = np.array(data)
plt.scatter(data[:,0], data[:,1])
plt.show()
