# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:57:09 2020

Author: David O'Callaghan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

BLACK = 0
RED = 1
GREEN = 2
YELLOW = 3

color_list = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'dimgrey']
n_colours = len(color_list)
cmap = colors.ListedColormap(color_list)
bounds = [i for i in range(n_colours + 1)]
norm = colors.BoundaryNorm(bounds, cmap.N)

class GridWorldEnvironment:
    
    # Encoding for colours
    black = 0
    red = 1
    green = 2
    yellow = 3
    
    def __init__(self, size, item_nums):
        
        # Grid dimensions
        self.grid_rows = size[0]
        self.grid_cols = size[1]
        
        # Number of each item
        self.num_red = item_nums[0]
        self.num_green = item_nums[1]
        self.num_yellow = item_nums[2]
        
        # Reset the env
        self.reset()
        
        # Define colours for plotting
        self.define_colours()
    
    def reset(self):
        """
        Initialises the grid.
        """
        # Initialise black grid
        self.grid = np.ones((self.grid_rows, self.grid_cols)) * BLACK
        
        # Get locations for each coloured item
        red_locs, green_locs, yellow_locs = self.randomise_items()
        
        # Set colours in grid
        for loc in red_locs:
            self.grid[loc] = RED
            
        for loc in green_locs:
            self.grid[loc] = GREEN
            
        for loc in yellow_locs:
            self.grid[loc] = YELLOW

    def randomise_items(self):
        """
        Randomise the item locations in the centre 4x4 square of the grid
        Returns:
            red_items    - List of coordinates for red item locations
            green_items  - List of coordinates for green item locations
            yellow_items - List of coordinates for yellow item locations
        """
        # initialise center area
        center_grid = []
        row_range = (self.grid_rows // 2 - 2, self.grid_rows // 2 + 2)
        col_range = (self.grid_cols // 2 - 2, self.grid_cols // 2 + 2)
        for i in range(*row_range):
            for j in range(*col_range):
                center_grid.append((i,j))
        
        # sample random locations for items
        num_items = self.num_red + self.num_green + self.num_yellow
        item_locations = random.sample(center_grid, num_items)
        
        # assign states to item colours
        red_items = item_locations[:self.num_red]
        green_items = item_locations[self.num_red:self.num_red+self.num_green]
        yellow_items = item_locations[self.num_red+self.num_green:]
        
        return red_items, green_items, yellow_items

    def define_colours(self):
        """
        Sets up the colours for plotting the gridworld
        """
        color_list = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'dimgrey']
        n_colours = len(color_list)
        self.cmap = colors.ListedColormap(color_list)
        bounds = [i for i in range(n_colours + 1)]
        self.norm = colors.BoundaryNorm(bounds, cmap.N)
        
    def display_grid(self):
        """
        Displays the current state of the gridworld
        """
        axes = plt.gca()
        axes.set_xlim(-0.5, self.grid_cols - 0.5)
        axes.set_ylim(-0.5, self.grid_rows - 0.5)
        axes.imshow(self.grid, cmap=self.cmap, norm=self.norm)
        axes.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=2)
        axes.set_xticks(np.arange(-.5, self.grid_cols, 1));
        axes.set_yticks(np.arange(-.5, self.grid_rows, 1));
        
    def step(self, action):
        pass
        #return obs, reward, done, info
    
    def render(self):
        pass
        
if __name__=="__main__":
    # Set seeds
    random.seed(23) # Remove for randomised grid
    np.random.seed(23)
    
    env = GridWorldEnvironment(size=(8,8), item_nums=(3,3,2))
    env.reset()
    env.display_grid()