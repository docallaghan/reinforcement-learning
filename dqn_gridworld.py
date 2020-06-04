# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:21:34 2020

Author: David O'Callaghan
"""
import numpy as np

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
        self.state = dict()
        self.init_grid()

    def init_grid(self):
        state = np.zeros(shape=(GRID_ROWS, GRID_COLS, N_CHANNELS))
        agent_loc, wall_loc, pit_loc, goal_loc = self.init_item_locations()
        
        state[(*agent_loc, AGENT_CHANNEL)] =  1 # agent start
        state[(*wall_loc, WALL_CHANNEL)] =  1  # wall location
        state[(*pit_loc, PIT_CHANNEL)] =  1  # pit location
        state[(*goal_loc, GOAL_CHANNEL)] =  1  # goal location
        
        # Store the locations of interest to avoid search
        self.state['state'] = state
        self.state['agent_loc'] = agent_loc
        self.state['wall_loc'] = wall_loc
        self.state['pit_loc'] = pit_loc
        self.state['goal_loc'] = goal_loc

    def init_item_locations(self):
        agent_loc = (0, 1)
        wall_loc = (2, 2)
        pit_loc = (1, 1)
        goal_loc = (3, 3)
        return agent_loc, wall_loc, pit_loc, goal_loc

    def step(self, action):
        
        cand_loc = (self.state['agent_loc'][0] + self.actions[action][0], 
                    self.state['agent_loc'][1] + self.actions[action][1])
    
        # Check if wall
        if cand_loc != self.state['wall_loc']:
            # Check if outside grid
            if ((cand_loc[0] <= GRID_ROWS-1 and cand_loc[0] >= 0) and
                (cand_loc[1] <= GRID_COLS-1 and cand_loc[1] >= 0)):
                # Erase old location
                self.state['state'][(*self.state['agent_loc'], AGENT_CHANNEL)] = 0
                # Write new location
                self.state['state'][(*cand_loc, AGENT_CHANNEL)] = 1
                self.state['agent_loc'] = cand_loc
        reward = self.get_reward(self.state)
        done = self.check_terminal_state(self.state)
        return self.state.copy(), reward, done 
    
    def get_reward(self, state):
        
        if self.state['agent_loc'] == self.state['pit_loc']:
            return -10 # Large Penalty
        elif self.state['agent_loc'] == self.state['goal_loc']:
            return 10 # Large Positive reward
        else:
            return -1 # Small Penalty
    
    def check_terminal_state(self, state):
        if ((self.state['agent_loc'] == self.state['pit_loc']) or
            (self.state['agent_loc'] == self.state['goal_loc'])):
            return True
        else:
            return False
    
    def display_grid(self):
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype='<U5')
        for i in range(0,GRID_ROWS):
            for j in range(0,GRID_COLS):
                grid[i,j] = ' '
        grid[self.state['agent_loc']] = 'A'
        grid[self.state['wall_loc']] = 'W'
        grid[self.state['pit_loc']] = '-'
        grid[self.state['goal_loc']] = '+'
        
        print(grid)
        
class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))] 
        
        self.alpha = 0.1 # Learning Rate
        self.gamma = 0.9 # Discount
        self.eps0 = 1.0 # Epsilon greedy init
        
        self.batch_size = 32
        self.replay_memory = deque(maxlen=2000)
        
        self.input_size = GRID_ROWS*GRID_COLS*N_CHANNELS
    
    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0,4)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values)
    
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        
        next_state, reward, done = self.env.step(action)
        next_state = next_state['state'].reshape((self.input_size,))
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done
    
    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def training_step(self):
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, 4) # 4 actions
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
    def train_nn(self, episodes, max_steps=20):

        self.model = keras.Sequential([
                keras.layers.Dense(16, input_shape=(self.input_size,), activation='elu'),
                keras.layers.Dense(16, activation='elu'),
                keras.layers.Dense(4)
                ])
        self.rewards = [] 
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        best_reward = 0
        for episode in range(episodes):
            # Get flattened initial state
            self.env.init_grid()
            obs = self.env.state['state'].copy()
            obs = obs.reshape((self.input_size,))
            
            episode_reward = 0
            for step in range(max_steps):
                eps = max(self.eps0 - episode / 500, 0.01) # decay epsilon
                obs, reward, done = self.play_one_step(obs, eps)
                episode_reward += reward
                if done:
                    break
            if episode_reward > best_reward:
                best_weights = self.model.get_weights()
                best_reward = episode_reward
            self.rewards.append(episode_reward)
            print("\rEpisode: {}, Reward: {}, eps: {:.3f}".format(episode, episode_reward, eps), end="")
            
            if episode > 50: # Wait for buffer to fill up a bit
                self.training_step()
        self.model.set_weights(best_weights)
            
    def plot_learning_curve(self):
        plt.plot(self.rewards)
        plt.xlabel('episode')
        plt.ylabel('reward per episode')
            
            
#        rms = RMSprop()
#        self.model.compile(loss='mse', optimizer=rms)
#        
#        gamma = 0.9 #since it may take several moves to goal, making gamma high
#        epsilon = 1
#        for i in range(episodes):
#            self.env.init_grid()
#            state = self.env.state['state']
#
#            #while game still in progress
#            while True:
#                #We are in state S
#                #Let's run our Q function on S to get Q values for all possible actions
#                qval = self.model.predict(state.reshape(1,input_size), batch_size=1)
#                if (random.random() < epsilon): #choose random action
#                    action = np.random.randint(0,4)
#                else: #choose best action from Q(s,a) values
#                    action = (np.argmax(qval))
#                #Take action, observe new state S' and reward
#                new_state, reward, done = self.env.step(action)
#                new_state = new_state['state']
#                #Get max_Q(S',a)
#                newQ = self.model.predict(new_state.reshape(1,64), batch_size=1)
#                maxQ = np.max(newQ)
#                y = np.zeros((1,4))
#                y[:] = qval[:]
#                if not done: #non-terminal state
#                    update = (reward + (gamma * maxQ))
#                else: #terminal state
#                    update = reward
#                y[0][action] = update #target output
#                
#                history = self.model.fit(state.reshape(1,64), y, batch_size=1, epochs=1, verbose=0)
#                loss = history.history['loss'][0]
#                print('\r', end='')
#                print("Game: {}  Loss: {}".format(i, loss), end='')
#                state = new_state
#                if done:
#                    break
#                clear_output(wait=True)
#            if epsilon > 0.1:
#                epsilon -= (1/episodes)
    
    def play_episode(self):
        self.env.init_grid()
        state = self.env.state['state']
    
        print("Initial State:")
        self.env.display_grid()
        #while game still in progress
        i=0
        while True:
            qval = self.model.predict(state.reshape(1,64))
            action = (np.argmax(qval)) #take action with highest Q-value
            print('Move #: %s; Taking action: %s' % (i, action))
            state, reward, done = self.env.step(action)
            state = state['state']
            env.display_grid()
            if done:
                print("Reward: %s" % (reward,))
                break
            i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
            if (i > 20):
                print("Game lost; too many moves.")
                break
            
env = Environment()
ag = Agent(env)
ag.train_nn(200)
ag.play_episode()

