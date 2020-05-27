# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:57:09 2020

Author: David O'Callaghan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import time
import random

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
    blue = 4
    
    # Reward Values
    transition_reward = -0.1
    red_reward = 1.0
    green_reward = 1.0
    yellow_reward = 1.0
    
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
        
        # Stores axis object later
        self.ax = None
        
    
    def reset(self):
        """
        Initialises the grid.
        """
        # Initialise black grid
        self.grid = np.ones((self.grid_rows, self.grid_cols)) * self.black
        
        # Initialise the agent location
        self.agent_location = (0,0)
        self.grid[self.agent_location] = self.blue
        
        # Get locations for each coloured item
        red_locs, green_locs, yellow_locs = self.randomise_items()
        
        # Set colours in grid
        for loc in red_locs:
            self.grid[loc] = self.red
            
        for loc in green_locs:
            self.grid[loc] = self.green
            
        for loc in yellow_locs:
            self.grid[loc] = self.yellow

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
        color_list = ['black', 'red', 'green', 'yellow', 'blue', 'pink']
        n_colours = len(color_list)
        self.cmap = colors.ListedColormap(color_list)
        bounds = [i for i in range(n_colours + 1)]
        self.norm = colors.BoundaryNorm(bounds, cmap.N)
        
    def initialise_grid_display(self):
        self.ax.set_xlim(-0.5, self.grid_cols - 0.5)
        self.ax.set_ylim(-0.5, self.grid_rows - 0.5)
        self.image = self.ax.imshow(self.grid, cmap=self.cmap, norm=self.norm)
        self.ax.grid(which='major', axis='both', linestyle='-', color='grey', 
                  linewidth=2)
        
        # Define ticks for cell borders
        self.ax.set_xticks(np.arange(-.5, self.grid_cols, 1))
        self.ax.set_yticks(np.arange(-.5, self.grid_rows, 1))
        
        # Disable tick labels
        self.ax.tick_params(labelleft=False, labelbottom=False)
        
        # Invert y-axis - (0,0) at top left instead of bottom left
        self.ax.set_ylim(self.ax.get_ylim()[::-1])
        
    def display_grid(self):
        """
        Displays the current state of the gridworld
        """  
        if self.ax == None:
            self.ax = plt.gca()
            self.initialise_grid_display()
        self.image.set_data(self.grid)
        plt.draw()
        plt.pause(0.001)
        
    def step(self, action):
        self.grid[self.agent_location] = self.black
        self.agent_location = self.get_next_agent_location(
                action, self.agent_location)
        reward = self.get_reward(self.agent_location)
        self.grid[self.agent_location] = self.blue
        
        # TODO: State should be more than agent location
        obs = self.agent_location
        return obs, reward
        #return obs, reward, done, info
    
    def get_next_agent_location(self, action, agent_location):
        if action == 4: # Stay
            return agent_location
        if action == 0: # Up                
            next_agent_location = (agent_location[0] - 1, agent_location[1])                
        elif action == 1: # Down
            next_agent_location = (agent_location[0] + 1, agent_location[1])
        elif action == 2: # Left
            next_agent_location = (agent_location[0], agent_location[1] - 1)
        elif action == 3: # Right
            next_agent_location = (agent_location[0], agent_location[1] + 1)
        
        # Check if next_state is within the grid limits
        if ((next_agent_location[0] >= 0) and 
            (next_agent_location[0] <= self.grid_rows - 1)):
            if ((next_agent_location[1] >= 0) and 
                (next_agent_location[1] <= self.grid_cols - 1)):                    
                    agent_location = next_agent_location
                    
        return agent_location
    
    def get_reward(self, agent_location):
        """
        Returns the reward for the state that the agent just transitioned to.
        """
        state_colour = self.grid[agent_location]
        if state_colour == self.black:
            return self.transition_reward
        
        elif state_colour == self.red:
            return self.red_reward
        
        elif state_colour == self.green:
            return self.green_reward
        
        elif state_colour == self.yellow:
            return self.yellow_reward      
    
    def render(self):
        pass
        
class Agent:
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
        self.actions = [0, 1, 2, 3] # up, down, left, right
        
        self.alpha = 0.5 # Learning Rate
        self.gamma = 0.9 # Discount
        self.eps = 0.1 # Epsillon greedy
    
    def q_learning(self, env, episodes):
        # Initial Q(s,a) to zeros
        self.q_values = np.zeros((*env.grid.shape, len(self.actions)))
        stats = []
        # Loop for each episode
        for n in range(episodes): 
            s = self.init_state # Initialise s
            rs = 0 # To store reward per episode
            
            step = 0
            while True: # Loop for each step in the episode
                
                # Choose action using policy derived from Q
                a = self.get_action(s)
                
                # Take action a and overve r and s'
                s_, r = env.step(a)
                rs += r
                
                # Update Q value
                qmax = self.get_qmax(s_)
                self.q_values[(*s,a)] += self.alpha * ( r + self.gamma*qmax - self.q_values[(*s,a)] )
                step += 1
                if step >= 30:
                    break
            stats.append([n, rs])
        self.stats = np.array(stats) # Save for plotting
        env.reset()
            
    def get_action(self, state):
        if np.random.rand() < self.eps:
            action =  np.random.choice(self.actions)
        else:
            q_values = self.q_values.copy()
            q_values_state = q_values[state[0], state[1],:]
            action = np.argmax(q_values_state)
        return action
    
    def get_qmax(self, state):
        q_values = self.q_values.copy()
        q_values_state = q_values[state[0], state[1],:]
        return np.max(q_values_state)
            
    def show_q_values(self):
        max_q_values = np.round(np.max(self.q_values, axis=2), 3)
        max_q_values = pd.DataFrame(max_q_values)
        print('Action Value Estimates:')
        print(max_q_values)
        print()
        
    def show_action_grid(self):
        action_grid = np.argmax(self.q_values, axis=2)
        actions = ['    U', '    D', '    L', '    R']
        action_grid_list = []
        for i in range(action_grid.shape[0]):
            row = []
            for j in range(action_grid.shape[1]):
                action_num = action_grid[i,j]
                row.append(actions[action_num])
            action_grid_list.append(row)
        action_grid_list = pd.DataFrame(action_grid_list)
        print('Actions:')
        print(action_grid_list)
        print()
        
    def plot_learning_curve(self):
        plt.plot(self.stats[:,0], self.stats[:,1], '-', alpha=0.8, linewidth=0.5)
        plt.xlabel('Episodes')
        plt.ylabel('Reward per Episode')
        plt.show()
        
    

        
if __name__=="__main__":
    # Set seeds
#    random.seed(23) # Remove for randomised grid
#    np.random.seed(23)
    
    env = GridWorldEnvironment(size=(8,8), item_nums=(3,3,2))
    #env.reset()
#    total_reward = 0
#    for _ in range(200):
#        env.display_grid()
#        plt.show()
#        action = random.choice([0,1,2,3])
#        obs, reward = env.step(action)
#        print(reward)
#        total_reward += reward
#    print(total_reward)

    # Instantiate Agent
    ag = Agent(init_state=(0,0))
    # Run Q Learning algorithm
    ag.q_learning(env, 100)
    
    ag.show_q_values()
    ag.show_action_grid()

    # Plotting
    ag.plot_learning_curve()