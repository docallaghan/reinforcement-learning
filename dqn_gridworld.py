# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:21:34 2020

Author: David O'Callaghan
"""
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

from collections import deque


AGENT_CHANNEL = 0
WALL_CHANNEL = 1
PIT_CHANNEL = 2
GOAL_CHANNEL = 3

GRID_ROWS = 4
GRID_COLS = 4
N_CHANNELS = 4

class Environment:
    
    # UP, DOWN, LEFT, RIGHT
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    def __init__(self):
        self.cell_locs = [(i,j) for i in range(GRID_ROWS) 
                          for j in range(GRID_COLS)]
        self.init_grid()

    def init_grid(self):
        self.state = np.zeros(shape=(GRID_ROWS, GRID_COLS, N_CHANNELS))
        agent_loc, wall_loc, pit_loc, goal_loc = self.init_item_locations()
        
        self.state[(*agent_loc, AGENT_CHANNEL)] =  1 # agent start
        self.state[(*wall_loc, WALL_CHANNEL)] =  1  # wall location
        self.state[(*pit_loc, PIT_CHANNEL)] =  1  # pit location
        self.state[(*goal_loc, GOAL_CHANNEL)] =  1  # goal location
        
        # Store the locations of interest to avoid search
        self.agent_loc = agent_loc
        self.wall_loc = wall_loc
        self.pit_loc = pit_loc
        self.goal_loc = goal_loc

    def init_item_locations(self):
        agent_loc, wall_loc, pit_loc, goal_loc = random.sample(self.cell_locs, 
                                                               k=4)
        return agent_loc, wall_loc, pit_loc, goal_loc
    
    def get_random_pair(self):
        return (np.random.randint(0, GRID_ROWS),
                np.random.randint(0, GRID_COLS))
    
    def reset(self):
        self.init_grid()
        state = self.get_current_state()
        return state
    
    def get_current_state(self):
        #state = self.state.copy()
        state = np.array([*self.agent_loc, 
                          *self.wall_loc, 
                          *self.pit_loc, 
                          *self.goal_loc])
        return state
    
    def set_current_state(self, state):
        self.state = state.copy()

    def step(self, action):
        
        # "Candidate" next location for the agent
        cand_loc = (self.agent_loc[0] + self.actions[action][0], 
                    self.agent_loc[1] + self.actions[action][1])
    
        # Check if wall
        if cand_loc != self.wall_loc:
            # Check if outside grid
            if ((cand_loc[0] <= GRID_ROWS-1 and cand_loc[0] >= 0) and
                (cand_loc[1] <= GRID_COLS-1 and cand_loc[1] >= 0)):
                # Erase old location
                self.state[(*self.agent_loc, AGENT_CHANNEL)] = 0
                # Write new location
                self.state[(*cand_loc, AGENT_CHANNEL)] = 1
                
                # Set the new location for the agent and the full env state
                self.agent_loc = cand_loc
        
        state = self.get_current_state()
        reward = self.get_reward()
        done = self.check_terminal_state()
        return state, reward, done 
    
    def get_reward(self):
        
        if self.agent_loc == self.pit_loc:
            return -10 # Large Penalty
        elif self.agent_loc == self.goal_loc:
            return 10 # Large Positive reward
        else:
            return -1 # Small Penalty
    
    def check_terminal_state(self):
        if ((self.agent_loc == self.pit_loc) or
            (self.agent_loc == self.goal_loc)):
            return True
        else:
            return False
    
    def display_grid(self):
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype='<U5')
        for i in range(0,GRID_ROWS):
            for j in range(0,GRID_COLS):
                grid[i,j] = ' '
        grid[self.agent_loc] = 'A'
        grid[self.wall_loc] = 'W'
        grid[self.pit_loc] = '-'
        grid[self.goal_loc] = '+'
        
        print(grid)
        
class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))] 
        
        self.alpha = 0.1 # Learning Rate
        self.gamma = 0.95 # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        #self.eps0 = 0.05 # Epsilon greedy init
        
        self.batch_size = 32
        self.replay_memory = deque(maxlen=2000)
        
        self.input_size = 8 #GRID_ROWS*GRID_COLS*N_CHANNELS
    
    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0,4)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values)
    
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done = self.env.step(action)
        next_state = next_state.reshape((self.input_size,))
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done
    
    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_memory), 
                                    size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def training_step(self):
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
        self.model = keras.Sequential([
                keras.layers.Dense(32, input_shape=(self.input_size,), 
                                   activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(4)
                ])

        self.optimizer = keras.optimizers.Adam(lr=1e-2)
        self.loss_fn = keras.losses.mean_squared_error
        
    def train_nn(self, episodes, max_steps=20):
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
            obs = obs.reshape((self.input_size,))
            
            episode_reward = 0
            for step in range(max_steps):
                eps = max(self.eps0 - episode / 500, 0.01) # decay epsilon
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
            
    def plot_learning_curve(self):
        y = np.array(self.rewards).reshape((-1,1))
        x = np.arange(1, len(y) + 1).reshape((-1,1))
        data = np.concatenate((x,y), axis=1)
        np.savetxt('rewards1.csv', data, delimiter=",")
        plt.plot(x, y)
        plt.xlabel('episode')
        plt.ylabel('reward per episode')
    
    def play_episode(self):
        self.env.init_grid()
        state = self.env.reset()
    
        print("Initial State:")
        self.env.display_grid()
        #while game still in progress
        i=0
        while True:
            qval = self.model.predict(state.reshape(1,self.input_size))
            action = (np.argmax(qval)) #take action with highest Q-value
            print('Move #: %s; Taking action: %s' % (i, action))
            state, reward, done = self.env.step(action)
            env.display_grid()
            if done:
                print("Reward: %s" % (reward,))
                break
            i += 1
            if (i > 20):
                print("Game lost; too many moves.")
                break

if __name__=="__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    env = Environment()
    ag = Agent(env)
    
    ag.train_nn(1000)
    ag.play_episode()
    ag.plot_learning_curve()

