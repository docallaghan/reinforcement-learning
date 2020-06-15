# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:23:17 2020

Author: David O'Callaghan
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

import tensorflow as tf
from tensorflow import keras

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script


DEBUG = True
# IMAGE = True

REPLAY_MEMORY_SIZE = 3000
BATCH_SIZE = 64

EPSILON_END = 0.05

COPY_TO_TARGET_EVERY = 1000 # Steps
START_TRAINING_AFTER = 50 # Episodes
MEAN_REWARD_EVERY = 10 # Episodes

GRID_ROWS = 8
GRID_COLS = 8

NUM_RED = 3
NUM_GREEN = 3
NUM_YELLOW = 2

RED_REWARD = 1
GREEN_REWARD = 1
YELLOW_REWARD = 1
TRANSITION_REWARD = -0.05


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
        Gets the current state of the environment.
        """
        state = self.grid.copy()
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
        if ((cand_loc[0] <= self.grid_rows-1 and cand_loc[0] >= 0) and
            (cand_loc[1] <= self.grid_cols-1 and cand_loc[1] >= 0)):
              
            # Erase old location
            self.grid[self.agent_loc] = np.zeros(3)
            # Write new location
            self.grid[cand_loc] = self.blue
            # Set the new location for the agent
            self.agent_loc = cand_loc
        
        rewards = self.__get_reward_vector()
        state = self.get_current_state()
        done = self.check_terminal_state()
        return state, rewards, done
    
    def __get_reward(self):
        """
        Returns the reward after an action has been taken. Also 
        """
        for i, item in enumerate(self.red_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.red_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up red!")
                return RED_REWARD
            
        for i, item in enumerate(self.green_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.green_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up green!")
                return GREEN_REWARD
            
        for i, item in enumerate(self.yellow_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.yellow_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up yellow!")
                return YELLOW_REWARD
        
        return TRANSITION_REWARD
    
    def __get_reward_vector(self):
        """
        Returns the reward after an action has been taken. Also 
        """
        # reward = [Green, Red, Yellow, Time] 
        
        for i, item in enumerate(self.green_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.green_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up green!")
                return [1, 0, 0, -1]
            
        for i, item in enumerate(self.red_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.red_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up red!")
                return [0, 1, 0, -1]
             
        for i, item in enumerate(self.yellow_items):
            if self.agent_loc == (item[0], item[1]) and item[2] == 1:
                self.yellow_items[i] = (item[0], item[1], 0)
                # if DEBUG:
                #     print("Picked up yellow!")
                return [0, 0, 1, -1]
        
        return [0, 0, 0, -1]
            
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


class ReplayMemory(deque):
    """
    Inherits from the 'deque' class to add a method called 'sample' for 
    sampling batches from the deque.
    """
    def sample(self, batch_size):
        """
        Sample a minibatch from the replay buffer.
        """
        # Random sample of indices
        indices = np.random.randint(len(self), 
                                    size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpach and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones, weightss = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


class RewardTracker:
    """
    Class for tracking mean rewards and storing all episode rewards for
    analysis.
    """
    def __init__(self, maxlen):
        self.moving_average = deque([-np.inf for _ in range(maxlen)], 
                                    maxlen=maxlen)
        self.maxlen = maxlen
        self.epsiode_rewards = []
        
    def __repr__(self):
        # For printing
        return self.moving_average.__repr__()
        
    def append(self, reward):
        self.moving_average.append(reward)
        self.epsiode_rewards.append(reward)
        
    def mean(self):
        return sum(self.moving_average) / self.maxlen
    
    def get_reward_data(self):
        episodes = np.array(
            [i for i in range(len(self.epsiode_rewards))]).reshape(-1,1)
        
        rewards = np.array(self.epsiode_rewards).reshape(-1,1)
        return np.concatenate((episodes, rewards), axis=1)


class WeightSpace:
    def __init__(self):
        # [Green, Red, Yellow, Time]
        self.distribution = [np.array([10, 10, 10, 10]), # Equal preferences
                             np.array([20, 5, 5, 10]) # Prefers green
                             ]
    def sample(self):
        return np.array(random.choice(self.distribution), dtype=np.float32)


class DQNAgent:
    
    def __init__(self, env, replay_memory):
        self.env = env
        self.actions = [i for i in range(len(env.actions))] 
        
        self.gamma = 0.95 # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        
        self.batch_size = BATCH_SIZE
        self.replay_memory = replay_memory
        
        self.input_size = self.env.get_current_state().shape
        self.output_size = len(self.actions)
        
        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())
        
        # Temporary
        self.pref_weights = [np.array([10, 10, 10]), # Equal preferences
                             np.array([20, 5, 5]) # Prefers green
                             ]
        
    def build_model(self):
        """
        Construct the DQN model.
        """
        # image of size 8x8 with 3 channels (RGB)
        image_input = keras.Input(shape=self.input_size)
        # preference weights
        weights_input = keras.Input(shape=(4,))
        
        # Convolutional Layers for image
        # - Define Layers
        conv2d_1 = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu')
        dropout_1 = keras.layers.Dropout(rate=0.2)
        conv2d_2 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')
        dropout_2 = keras.layers.Dropout(rate=0.2)
        flatten = keras.layers.Flatten()
        # - Define Architecture
        x = conv2d_1(image_input)
        x = dropout_1(x)
        x = conv2d_2(x)
        x = dropout_2(x)
        image_output = flatten(x)
        
        # Dense layers for weight input concatenated with conv layers output
        # - Define Layers
        dense = keras.layers.Dense(32, activation='relu')
        output = keras.layers.Dense(self.output_size)
        # - Define Architecture
        dense_input = keras.layers.concatenate([image_output, weights_input])
        x = dense(dense_input)
        outputs = output(x)  
        
        # Build full model
        model = keras.Model(inputs=[image_input, weights_input], outputs=outputs)
        
        # Define optimizer and loss function
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        
        return model
        
    # def build_model(self):
    #     """
    #     Construct the DQN model.
    #     """

    #     if IMAGE:
    #         model = keras.Sequential([
    #             keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.env.grid.shape),
    #             keras.layers.Dropout(0.2),
    #             keras.layers.Conv2D(16, (3, 3), activation='relu'),
    #             keras.layers.Dropout(0.2),
    #             keras.layers.Flatten(),
    #             keras.layers.Dense(32, activation='relu'),
    #             keras.layers.Dense(4)
    #             ])
    #     else:
    #         model = keras.Sequential([
    #             keras.layers.Dense(128, input_shape=(self.input_size,), 
    #                                activation='relu'),
    #             keras.layers.Dense(64, activation='relu'),
    #             keras.layers.Dense(4)
    #             ])

    #     self.optimizer = keras.optimizers.Adam(lr=1e-3)
    #     self.loss_fn = keras.losses.mean_squared_error
        
    #     #model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
    #     return model
    
    def epsilon_greedy_policy(self, state, epsilon, weights):
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model.predict([state[np.newaxis], weights[np.newaxis]])
            return np.argmax(Q_values)
    
    def play_one_step(self, state, epsilon, weights):
        """
        Play one action using the DQN and store S A R S' in replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        action = self.epsilon_greedy_policy(state, epsilon, weights)
        
        next_state, rewards, done = self.env.step(action)
        reward = np.dot(rewards, weights) # Linear scalarisation
        self.replay_memory.append((state, action, reward, next_state, done, weights))
        return next_state, reward, done
    
    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from: 
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences
        
        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model.predict([next_states, weightss])
        
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, 4) # 4 actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([next_states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, 
                                     keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, 
                                           self.model.trainable_variables))
        
    def train_model(self, episodes, reward_tracker, weight_space):
        """
        Train the network over a range of episodes.
        """
        best_reward = -np.inf
        steps = 0
        
        for episode in range(episodes):
            weights = weight_space.sample()
            # Decay epsilon
            eps = max(self.eps0 - episode * EPSILON_DECAY, EPSILON_END)
            
            # Reset env
            state = self.env.reset()
            
            episode_reward = 0
            while True:
                
                #eps = self.eps0
                state, reward, done = self.play_one_step(state, eps, weights)
                steps += 1
                episode_reward += reward
                if done:
                    break
                
                # Copy weights from main model to target model
                if steps % COPY_TO_TARGET_EVERY == 0:
                    if DEBUG:
                        print(f'\n\n{steps}: Copying to target\n\n')
                    self.target_model.set_weights(self.model.get_weights())
                        
            reward_tracker.append(episode_reward)
            avg_reward = reward_tracker.mean()
            if avg_reward > best_reward:
                #best_weights = self.model.get_weights()
                best_reward = avg_reward
            
            print("\rEpisode: {}, Reward: {}, Avg Reward {}, eps: {:.3f}".format(
                episode, episode_reward, avg_reward, eps), end="")
            
            if episode > START_TRAINING_AFTER: # Wait for buffer to fill up a bit
                self.training_step()
        # self.model.set_weights(best_weights)
        self.reward_data = reward_tracker.get_reward_data()
        self.model.save(MODEL_PATH)
    
    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
            
    def plot_learning_curve(self, image_path=None, csv_path=None):
        """
        Plot the rewards per episode collected during training
        """
        fig, ax = plt.subplots()
        x = self.reward_data[:,0]
        y = self.reward_data[:,1]
        
        if csv_path:
            np.savetxt(csv_path, self.reward_data, delimiter=",")
        ax.plot(x, y)
        ax.set_xlabel('episode')
        ax.set_ylabel('reward per episode')
        if image_path:
            fig.savefig(image_path)
    
    def play_episode(self, weights):
        """
        Play one episode using the DQN and display the grid image at each step.
        """
        state = self.env.reset()
    
        print("Initial State:")
        self.env.show(boundaries=True)
        i = 0
        episode_reward = 0
        while True:
            i += 1
            #qval = self.model.predict(state.reshape(1,self.input_size))
            action = self.epsilon_greedy_policy(state, 0.05, weights)
            #action = (np.argmax(qval)) #take action with highest Q-value
            print('Move #: %s; Taking action: %s' % (i, action))
            state, rewards, done = self.env.step(action)
            reward = np.dot(rewards, weights)
            episode_reward += reward
            self.env.show(boundaries=True)
            if done:
                print(f'Reward: {episode_reward}')
                break

def get_command_line_args(parser):
    parser.add_argument("-i", "--ID",
                        help="File ID for output")
    parser.add_argument("-s", "--SEED", type=int,
                        help="Random seed")
    parser.add_argument("-e", "--EPISODES", type=int,
                        help="Number of episodes")
    parser.add_argument("-r", "--RESTARTS", type=int,
                        help="Number of exploaration restarts")
    return parser.parse_args()
    

if __name__ == '__main__':
    # Get arguments from command line ...
    parser = argparse.ArgumentParser()
    args = get_command_line_args(parser)
    SEED = args.SEED
    PATH_ID = args.ID
    
    TRAINING_EPISODES = args.EPISODES
    EXPLORATION_RESTARTS = args.RESTARTS
    
    IMAGE_PATH = f'plots/reward_plot_{PATH_ID}.png'
    CSV_PATH = f'plots/reward_data_{PATH_ID}.csv'
    MODEL_PATH = f'models/dqn_model_{PATH_ID}.h5'
    
    EPSILON_DECAY = 1 / (args.EPISODES / (args.RESTARTS + 1))
    
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    # For timing the script
    start_time = datetime.now()
    
    # Instantiate environment
    item_env = ItemGatheringGridworld()
    
    # Initialise Replay Memory
    replay_mem = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
    
    # Initialise Reward Tracker
    reward_track = RewardTracker(maxlen=MEAN_REWARD_EVERY)
    
    # Initialise Preference Weight Space
    pref_space = WeightSpace()
    
    # Instantiate agent (pass in environment)
    dqn_ag = DQNAgent(item_env, replay_mem)
    
    # Train agent
    dqn_ag.train_model(TRAINING_EPISODES, reward_track, pref_space)
    for _ in range(EXPLORATION_RESTARTS):
        dqn_ag.train_model(TRAINING_EPISODES, reward_track, pref_space)
        
    dqn_ag.plot_learning_curve(image_path=IMAGE_PATH, 
                               csv_path=CSV_PATH)
    
    # Play episode with learned DQN
    weights = pref_space.sample()
    print(f'\n{weights}\n')
    dqn_ag.play_episode(weights)

    run_time = datetime.now() - start_time
    print(f'Run time: {run_time} s')
    
