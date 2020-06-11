# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:50:22 2020

Author: David O'Callaghan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from deep_sea_treasure import DeepSeaTreasureEnvironment

def is_pareto_efficient(costs):
    """
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    
    Find the pareto-efficient points
    
    :param costs: An (n_points, n_costs) array
    
    :return: is_efficient (n_points, ) boolean array, indicating whether each 
    point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

if __name__ == '__main__':
    with open('dst_results.pkl', 'rb') as f:
        stats_dict = pickle.load(f)
    
    data = []
    for key in stats_dict:
        Q_values = stats_dict[key][1]
        env = DeepSeaTreasureEnvironment()
        state = env.reset()
        r = [0,0]
        while True:
            action = np.argmax(Q_values[(*state,)])
            state, rewards, done = env.step(action)
            r[0] += rewards[0]
            r[1] += rewards[1]
            if done:
                print(f'Finished {key}')
                print(f'{r}\n')
                data.append(r)
                break
            
    data = np.array(data)
    # Filter the pareto front
    est_pareto = data[is_pareto_efficient(data)]
    true_pareto = np.array([[-1,-3,-5,-7,-8,-9,-13,-14,-17,-19],
                            [1,34,58,78,86,92,112,116,122,124]]).T
    # Plot
    style.use("default")
    fig, ax = plt.subplots()
    ax.scatter(true_pareto[:,0], true_pareto[:,1], s=80, c='r', marker='x')
    ax.scatter(est_pareto[:,0], est_pareto[:,1], s=80, c='b', marker='+')
    
    ax.set_xlabel('Time Penalty')
    ax.set_ylabel('Treasure Value')
    ax.set_title('Deep Sea Treasure')
    ax.legend(['True Pareto Front', 'Found Pareto Front'])
    ax.grid(True)
    plt.show()
