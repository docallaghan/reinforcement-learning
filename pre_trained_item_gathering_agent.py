# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:46:22 2020

Author: David O'Callaghan
"""

from item_gathering_gridworld import ItemGatheringGridworld
from item_gathering_gridworld import DQNAgent

import numpy as np
import random
import tensorflow as tf

from datetime import datetime # Used for timing script
import time

SEED = 42
PATH_ID = '001'
MODEL_PATH = f'models/dqn_model_{PATH_ID}.h5'


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    # For timing the script
    start_time = datetime.now()
    
    # Instantiate environment
    item_env = ItemGatheringGridworld()
    
    # Instantiate agent (pass in environment)
    dqn_ag = DQNAgent(item_env)
    
    # Load pre-trained agent
    dqn_ag.load_model(MODEL_PATH)
    
    # Play episode with learned DQN
    for _ in range(10):
        dqn_ag.play_episode()
        time.sleep(0.5)

    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
    