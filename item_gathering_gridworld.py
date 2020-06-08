# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:23:17 2020

Author: David O'Callaghan
"""

import numpy as np
import matplotlib.pyplot as plt
import random

DEBUG = True

GRID_ROWS = 8
GRID_COLS = 8

NUM_RED = 3
NUM_GREEN = 3
NUM_YELLOW = 2

RED_REWARD = 10
GREEN_REWARD = 11
YELLOW_REWARD = 12
TRANSITION_REWARD = -1

class ItemGatheringGridworld:
    
    # Colours
    pink = np.array([255, 0, 255]) / 255
    blue = np.array([0, 0, 255]) / 255
    yellow = np.array([255, 255, 0]) / 255
    red = np.array([255, 0, 0]) / 255
    green = np.array([0, 255, 0]) / 255
    
    # UP, DOWN, LEFT, RIGHT
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    def __init__(self):
        
        # Grid dimensions
        self.grid_rows = GRID_ROWS
        self.grid_cols = GRID_COLS
        
        # Number of each item
        self.num_red = NUM_RED
        self.num_green = NUM_GREEN
        self.num_yellow = NUM_YELLOW
        
        # Initialise the environment
        self.reset()
        
        # Stores axis object later
        self.ax = None
        
        
    def inititialise_grid(self):
        """
        Initialises the grid.
        """
        
        # Initialise the agent location
        self.agent_loc = (0, 0)
        
        # All possible item locations
        row_range = (self.grid_rows // 2 - 2, self.grid_rows // 2 + 2)
        col_range = (self.grid_cols // 2 - 2, self.grid_cols // 2 + 2)
        item_locs = [(i,j,1) for i in range(*row_range) 
                            for j in range(*col_range)]
        
        # Sample random locations for items
        num_items = self.num_red + self.num_green + self.num_yellow
        item_locations = random.sample(item_locs, num_items)
        
        # Assign states to item colours
        self.red_items = item_locations[:self.num_red]
        self.green_items = item_locations[self.num_red:self.num_red+self.num_green]
        self.yellow_items = item_locations[self.num_red+self.num_green:]
        
        # Initialise grid
        self.grid = np.zeros((self.grid_rows, self.grid_cols, 3))
        
        # Place agent
        self.grid[self.agent_loc] = self.blue
        
        # Place items
        for items, colour in zip([self.red_items, self.green_items, self.yellow_items],
                                [self.red, self.green, self.yellow]):
            for item in items:
                loc = (item[0], item[1]) # x and y coords
                self.grid[loc] = colour
                
    def reset(self):
        """
        Initialises the grid and returns the current state.
        """
        self.n_steps = 0
        self.inititialise_grid()
        state = self.get_current_state()        
        return state
    
    def get_current_state(self):
        """
        Gets the current state of the environment. Returns a 1D NumPy array
        where the first 2 elements are the agent x and y locations and each
        subsequent set of 3 elements are the item x, y coords and whether the
        items has is still present.
        """
        state = []
        for elem in self.agent_loc:
            state.append(elem)
        
        for items, colour in zip([self.red_items, self.green_items, self.yellow_items],
                                [self.red, self.green, self.yellow]):
            for item in items:
                for elem in item: # (x, y, item-present)
                    state.append(elem)
                
        return np.array(state)
    
    def step(self, action):
        """
        Transition the environment through the input action
        """
        self.n_steps += 1
        # "Candidate" next location for the agent
        cand_loc = (self.agent_loc[0] + self.actions[action][0], 
                    self.agent_loc[1] + self.actions[action][1])
    

        # Check if outside grid
        if ((cand_loc[0] <= GRID_ROWS-1 and cand_loc[0] >= 0) and
            (cand_loc[1] <= GRID_COLS-1 and cand_loc[1] >= 0)):
              
            # Erase old location
            self.grid[self.agent_loc] = np.zeros(3)
            # Write new location
            self.grid[cand_loc] = self.blue
            # Set the new location for the agent
            self.agent_loc = cand_loc
        
        reward = self.__get_reward()
        state = self.get_current_state()
        done = self.check_terminal_state()
        return state, reward, done
    
    def __get_reward(self):
        """
        Returns the reward after an action has been taken. Also 
        """
        for i, item in enumerate(self.red_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.red_items[i] = (item[0], item[1], 0)
                if DEBUG:
                    print("Picked up red!")
                return RED_REWARD
            
        for i, item in enumerate(self.green_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.green_items[i] = (item[0], item[1], 0)
                if DEBUG:
                    print("Picked up green!")
                return GREEN_REWARD
            
        for i, item in enumerate(self.yellow_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.yellow_items[i] = (item[0], item[1], 0)
                if DEBUG:
                    print("Picked up yellow!")
                return YELLOW_REWARD
        
        return TRANSITION_REWARD
            
    def check_terminal_state(self):
        """
        Checks if the max number of states has been exceeded. Retruns True if
        it has and False otherwise.
        """
        return self.n_steps >= 50
    
    def __initialise_grid_display(self, boundaries):
        # Set axis limits
        self.ax.set_xlim(-0.5, self.grid_cols - 0.5)
        self.ax.set_ylim(-0.5, self.grid_rows - 0.5)
        
        # Show boundaries between grid cells
        if boundaries:
            self.ax.grid(which='major', axis='both', linestyle='-', color='grey', 
                    linewidth=2)
        
        # Define ticks for cell borders
        self.ax.set_xticks(np.arange(-.5, self.grid_cols, 1))
        self.ax.set_yticks(np.arange(-.5, self.grid_rows, 1))
        
        # Disable tick labels
        self.ax.tick_params(labelleft=False, labelbottom=False)
        
        # Invert y-axis - (0,0) at top left instead of bottom left
        self.ax.set_ylim(self.ax.get_ylim()[::-1])
        
        
        # Display image
        self.image = self.ax.imshow(self.grid)
        
    def show(self, boundaries=False):
        """
        Displays the gridworld image.
        """  
        if self.ax == None:
            _, self.ax = plt.subplots()
            self.__initialise_grid_display(boundaries)
        self.image.set_data(self.grid)
        plt.draw()
        plt.pause(0.05)

        
if __name__ == '__main__':
    np.random.seed(23)
    env = ItemGatheringGridworld()
    env.show(boundaries=True)
    print(env.get_current_state())
    print()
    
    for i in range(100):
        state, reward, done = env.step(random.choice([0,1,2,3]))
        print(i, reward)
        env.show(True)
        if done:
            break
        
    plt.show()
    
