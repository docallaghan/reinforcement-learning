# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:33:48 2020

Author: David O'Callaghan

This code is based off the blogpost at http://outlace.com/rlpart3.html
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop

from IPython.display import clear_output
import random


GRID_N = 4

AGENT = np.array([0,0,0,1])
WALL = np.array([0,0,1,0])
PIT = np.array([0,1,0,0])
GOAL = np.array([1,0,0,0])

def rand_pair(start, end):
    """ Generates a random pair"""
    a = np.random.randint(start, end)
    b = np.random.randint(start, end)
    return a, b

def find_loc(state, obj):
    """ Finds an array in the depth dimension of the grid"""
    for i in range(GRID_N):
        for j in range(GRID_N):
            if (state[i,j] == obj).all():
                return i, j
            
def init_grid():
    """Easy Grid - All items placed determinisitcally"""
    state = np.zeros(shape=(GRID_N,GRID_N,4))
    state[0,1] = AGENT # agent start
    state[2,2] = WALL  # wall location
    state[1,1] = PIT   # pit location
    state[3,3] = GOAL  # goal location
    return state

def init_grid_player():
    """Medium Grid - Agent location randomised all others deterministic"""
    state = np.zeros(shape=(GRID_N,GRID_N,4))
    state[rand_pair(0, GRID_N)] = AGENT # agent start
    state[2,2] = WALL  # wall location
    state[1,1] = PIT   # pit location
    state[3,3] = GOAL  # goal location
    
    # Make sure the grid is valid (due to randomisation)
    a = find_loc(state, AGENT) #find grid position of agent
    w = find_loc(state, WALL) #find wall
    p = find_loc(state, PIT) #find pit
    g = find_loc(state, GOAL) #find goal
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return init_grid_player()
    return state

def init_grid_rand():
    """Medium Grid - Agent location randomised all others deterministic"""
    state = np.zeros(shape=(GRID_N,GRID_N,4))
    state[rand_pair(0, GRID_N)] = AGENT # agent start
    state[rand_pair(0, GRID_N)] = WALL  # wall location
    state[rand_pair(0, GRID_N)] = PIT   # pit location
    state[rand_pair(0, GRID_N)] = GOAL  # goal location
    
    # Make sure the grid is valid (due to randomisation)
    a = find_loc(state, AGENT) #find grid position of agent
    w = find_loc(state, WALL) #find wall
    p = find_loc(state, PIT) #find pit
    g = find_loc(state, GOAL) #find goal
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return init_grid_rand()
    return state

def make_move(state, action):
    
    agent_loc = find_loc(state, AGENT)
    wall = find_loc(state, WALL)
    pit = find_loc(state, PIT)
    goal = find_loc(state, GOAL)
    
    state = np.zeros(shape=(GRID_N,GRID_N,4))
    
    # UP, DOWN, LEFT, RIGHT
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    new_loc = (agent_loc[0] + actions[action][0], 
               agent_loc[1] + actions[action][1])

    # Take action
    if new_loc != wall:
        if ((np.array(new_loc) <= (GRID_N-1,GRID_N-1)).all() and 
            (np.array(new_loc) >= (0,0)).all()):
            state[new_loc[0], new_loc[1], 3] = 1 # Agent channel is 3
    
    # See if action was successful
    new_agent_loc = find_loc(state, np.array([0,0,0,1]))
    if (not new_agent_loc):
        state[agent_loc] = np.array([0,0,0,1])
    
    #re-place wall
    state[wall][2] = 1 # Wall channel is 2    
    #re-place pit
    state[pit][1] = 1 # Pit channel is 1
    #re-place goal
    state[goal][0] = 1 # Goal channel is 0

    return state

def get_loc(state, level):
    """Finds an array in the depth dimension of the grid"""
    for i in range(GRID_N):
        for j in range(GRID_N):
            if (state[i,j][level] == 1):
                return i, j
            
def get_reward(state):
    player_loc = get_loc(state, 3) # agent channel is 3
    pit = get_loc(state, 1) # pit channel is 1
    goal = get_loc(state, 0) # goal channel is 0
    if player_loc == pit:
        return -10 # Large Penalty
    elif player_loc == goal:
        return 10 # Large Positive reward
    else:
        return -1 # Small Penalty
    
def disp_grid(state):
    grid = np.zeros((4,4), dtype='<U5')
    player_loc = find_loc(state, np.array([0,0,0,1]))
    wall = find_loc(state, np.array([0,0,1,0]))
    goal = find_loc(state, np.array([1,0,0,0]))
    pit = find_loc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '
            
    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit
    
    return grid

def testAlgo(init=0):
    i = 0
    if init==0:
        state = init_grid()
    elif init==1:
        state = init_grid_player()
    elif init==2:
        state = init_grid_rand()

    print("Initial State:")
    print(disp_grid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = make_move(state, action)
        print(disp_grid(state))
        reward = get_reward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break

if __name__ == "__main__":
    state = init_grid_rand()
    print(disp_grid(state))
    state = make_move(state, 3)
    print(disp_grid(state))
    
    input_size = GRID_N*GRID_N*4 # 64
    model = Sequential([
            Dense(164, input_shape=(input_size,), activation='relu'),
            Dense(150, activation='relu'),
            Dense(4, activation='linear')
            ])

    rms = RMSprop()
#    model.compile(loss='mse', optimizer=rms)
#    
#    epochs = 1000
#    gamma = 0.9 #since it may take several moves to goal, making gamma high
#    epsilon = 1
#    for i in range(epochs):
#        
#        state = init_grid()
#        status = 1
#        #while game still in progress
#        while(status == 1):
#            #We are in state S
#            #Let's run our Q function on S to get Q values for all possible actions
#            qval = model.predict(state.reshape(1,64), batch_size=1)
#            if (random.random() < epsilon): #choose random action
#                action = np.random.randint(0,4)
#            else: #choose best action from Q(s,a) values
#                action = (np.argmax(qval))
#            #Take action, observe new state S'
#            new_state = make_move(state, action)
#            #Observe reward
#            reward = get_reward(new_state)
#            #Get max_Q(S',a)
#            newQ = model.predict(new_state.reshape(1,64), batch_size=1)
#            maxQ = np.max(newQ)
#            y = np.zeros((1,4))
#            y[:] = qval[:]
#            if reward == -1: #non-terminal state
#                update = (reward + (gamma * maxQ))
#            else: #terminal state
#                update = reward
#            y[0][action] = update #target output
#            
#            history = model.fit(state.reshape(1,64), y, batch_size=1, epochs=1, verbose=0)
#            loss = history.history['loss'][0]
#            print('\r', end='')
#            print("Game: {}  Loss: {}".format(i, loss), end='')
#            state = new_state
#            if reward != -1:
#                status = 0
#            clear_output(wait=True)
#        if epsilon > 0.1:
#            epsilon -= (1/epochs)
#    
#    testAlgo(init=0)
    
    
    model.compile(loss='mse', optimizer=rms)#reset weights of neural network
    epochs = 3000
    gamma = 0.975
    epsilon = 1
    batchSize = 40
    buffer = 80
    replay = []
    #stores tuples of (S, A, R, S')
    h = 0
    for i in range(epochs):
        
        state = init_grid_player() #using the harder state initialization function
        status = 1
        #while game still in progress
        while(status == 1):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1,64), batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'
            new_state = make_move(state, action)
            #Observe reward
            reward = get_reward(new_state)
            
            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                #randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
                    newQ = model.predict(new_state.reshape(1,64), batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1,4))
                    y[:] = old_qval[:]
                    if reward == -1: #non-terminal state
                        update = (reward + (gamma * maxQ))
                    else: #terminal state
                        update = reward
                    y[0][action] = update
                    X_train.append(old_state.reshape(64,))
                    y_train.append(y.reshape(4,))
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                print("Game #: %s" % (i,))
                model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=2)
                state = new_state
            if reward != -1: #if reached terminal state, update game status
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1: #decrement epsilon over time
            epsilon -= (1/epochs)
    
    
