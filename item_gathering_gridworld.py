# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:23:17 2020

Author: David O'Callaghan
"""

import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script


DEBUG = False
IMAGE = True
SEED = 42

PATH_ID = '001'
IMAGE_PATH = f'plots/reward_plot_{PATH_ID}.png'
CSV_PATH = f'plots/reward_data_{PATH_ID}.csv'
MODEL_PATH = f'models/dqn_model_{PATH_ID}.h5'

TRAINING_EPISODES = 3000

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
        if IMAGE:
            state = self.grid.copy()
        else:
            state = []
            for elem in self.agent_loc:
                state.append(elem)
            
            for items, colour in zip([self.red_items, self.green_items, self.yellow_items],
                                    [self.red, self.green, self.yellow]):
                for item in items:
                    for elem in item: # (x, y, item-present)
                        state.append(elem)
            state = np.array(state)

        return state
    
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
        all_items_collected = not any([item[2] for item in (
            *self.red_items, *self.green_items, *self.yellow_items)])

            
        return self.n_steps >= 50 or all_items_collected
    
    def __initialise_grid_display(self, boundaries):
        """
        Set up the plot objects for displaying an the gridworld image.
        """
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


class DQNAgent:
    
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))] 
        
        self.alpha = 0.1 # Learning Rate
        self.gamma = 0.95 # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        #self.eps0 = 0.1 # Epsilon greedy init
        
        self.batch_size = 64
        self.replay_memory = deque(maxlen=2000)
        
        if IMAGE:
            self.input_size = item_env.get_current_state().shape
        else:
            self.input_size = len(self.env.get_current_state()) 
    
    def epsilon_greedy_policy(self, state, epsilon):
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values)
    
    def play_one_step(self, state, epsilon):
        """
        Play one action using the DQN and store S A R S' in replay buffer.
        """
        action = self.epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done = self.env.step(action)
        #next_state = next_state.reshape((self.input_size,))
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done
    
    def sample_experiences(self):
        """
        Sample a batch from the replay buffer.
        """
        indices = np.random.randint(len(self.replay_memory), 
                                    size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones = experiences
        
        # Compute target Q values from 'next_states'
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, 4) # 4 actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, 
                                     keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, 
                                           self.model.trainable_variables))
        
    def build_model(self):
        """
        Construct the DQN model.
        """

        if IMAGE:
            self.model = keras.Sequential([
                keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.env.grid.shape),
                #keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(16, (3, 3), activation='relu'),
                #keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(4)
                ])
        else:
            self.model = keras.Sequential([
                keras.layers.Dense(128, input_shape=(self.input_size,), 
                                   activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(4)
                ])


        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        
    def train_nn(self, episodes):
        """
        Train the network over a range of episodes.
        """
        self.build_model()
        
        self.rewards = [] 
        best_reward = -1000
        n_rewards = 10
        reward_list = deque([best_reward for _ in range(n_rewards)], maxlen=n_rewards)
        for episode in range(episodes):
            # Get flattened initial state
            #self.env.init_grid()
            #obs = self.env.state['state'].copy()
            obs = self.env.reset()
            #obs = obs.reshape((self.input_size,))
            
            episode_reward = 0
            while True:
                eps = max(self.eps0 - episode / 1000, 0.05) # decay epsilon
                #eps = self.eps0
                obs, reward, done = self.play_one_step(obs, eps)
                episode_reward += reward
                if done:
                    break
                
            self.rewards.append(episode_reward)
            reward_list.append(episode_reward)
            avg_reward = sum(reward_list) / n_rewards
            if avg_reward > best_reward:
                best_weights = self.model.get_weights()
                best_reward = avg_reward
            
            print("\rEpisode: {}, Reward: {}, Avg Reward {}, eps: {:.3f}".format(
                episode, episode_reward, avg_reward, eps), end="")
            
            if episode > 50: # Wait for buffer to fill up a bit
                self.training_step()
        self.model.set_weights(best_weights)
        self.model.save(MODEL_PATH)
            
    def plot_learning_curve(self, image_path=None, csv_path=None):
        """
        Plot the rewards per episode collected during training
        """
        fig, ax = plt.subplots()
        y = np.array(self.rewards).reshape((-1,1))
        x = np.arange(1, len(y) + 1).reshape((-1,1))
        if csv_path:
            data = np.concatenate((x,y), axis=1)
            np.savetxt(csv_path, data, delimiter=",")
        ax.plot(x, y)
        ax.set_xlabel('episode')
        ax.set_ylabel('reward per episode')
        if image_path:
            fig.savefig(image_path)
    
    def play_episode(self):
        """
        Play one episode using the DQN and display the grid image at each step.
        """
        state = self.env.reset()
    
        print("Initial State:")
        self.env.show()
        i = 0
        rewards = 0
        while True:
            i += 1
            #qval = self.model.predict(state.reshape(1,self.input_size))
            action = self.epsilon_greedy_policy(state, 0.05)
            #action = (np.argmax(qval)) #take action with highest Q-value
            print('Move #: %s; Taking action: %s' % (i, action))
            state, reward, done = self.env.step(action)
            rewards += reward
            self.env.show()
            if done:
                print("Reward: %s" % (rewards,))
                break
            

if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # For timing the script
    start_time = datetime.now()
    
    # Instantiate environment
    item_env = ItemGatheringGridworld()
    
    # Instantiate agent (pass in environment)
    dqn_ag = DQNAgent(item_env)
    
    # Train agent
    dqn_ag.train_nn(TRAINING_EPISODES)
    dqn_ag.plot_learning_curve(image_path=IMAGE_PATH, 
                               csv_path=CSV_PATH)
    
    # Play episode with learned DQN
    dqn_ag.play_episode()

    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
    
