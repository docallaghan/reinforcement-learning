# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:33:48 2020

Author: David O'Callaghan
"""

import numpy as np

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
        if ((np.array(new_loc) <= (GRID_N,GRID_N)).all() and 
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

if __name__ == "__main__":
    state = init_grid_rand()
    print(disp_grid(state))
    state = make_move(state, 3)
    print(disp_grid(state))
    
    