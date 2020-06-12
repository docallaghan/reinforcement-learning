# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:53:15 2020
Author: David O'Callaghan
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script


SEED = 42
WEIGHT_IN = True


class DeepSeaTreasureEnvironment:
    
    grid_rows = 11
    grid_cols = 10
    
    depths = [1, 2, 3, 4, 4, 4, 7, 7, 9, 10]
    #treasure = [0.5, 28, 52, 73, 82, 90, 115, 120, 134, 143]
    treasure = [1, 34, 58, 78, 86, 92, 112, 116, 122, 124]
    
    # UP, DOWN, LEFT, RIGHT
    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    
    def __init__(self):
        self.reset()
        self.forbidden_states = self.__get_forbidden_states()
        self.treasure_locations = self.__get_treasure_locations()
    
    def __get_forbidden_states(self):
        forbidden_states = [(i, j) for j in range(self.grid_cols) 
                            for i in range(self.depths[j]+1, self.grid_rows)]
        return forbidden_states
    
    def __get_treasure_locations(self):
        treasure_locations = [(i, j) for j, i in enumerate(self.depths)]
        return treasure_locations
            
    def reset(self):
        self.n_steps = 0
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """
        Transition the environment through the input action
        """
        self.n_steps += 1
        # "Candidate" next location for the agent
        cand_loc = (self.state[0] + self.actions[action][0], 
                    self.state[1] + self.actions[action][1])
    

        # Check if forbidden state
        if ((cand_loc[0] <= self.grid_rows-1 and cand_loc[0] >= 0) and
            (cand_loc[1] <= self.grid_cols-1 and cand_loc[1] >= 0) and
            (cand_loc not in self.forbidden_states)):
            # Set new state
            self.state = cand_loc
        
        rewards = self.get_rewards()
        state = self.state
        done = self.check_terminal_state()
        return state, rewards, done
    
    def get_rewards(self):
        rewards = [-1, 0] # (time_penalty, treasure_reward)
        if self.state in self.treasure_locations:
            rewards[1] = self.treasure[self.state[1]]
        return tuple(rewards)
    
    def check_terminal_state(self):
        return (self.state in self.treasure_locations) or (self.n_steps > 200)


class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [i for i in range(len(env.actions))]
    
    def epsillon_greedy_policy(self, state, epsillon):
        if np.random.rand() < epsillon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q_values[state])
        
    def scalarise(self, rewards, weights):
        
        rewards = np.array(rewards)
        return np.dot(rewards, weights)
    
    def initialise_q_values(self):
        # Q_values = np.random.randint(0, 125, size=(self.env.grid_rows, 
        #                                            self.env.grid_cols, 
        #                                            len(self.env.actions)))
        Q_values = np.random.rand(self.env.grid_rows,
                                  self.env.grid_cols,
                                  len(self.env.actions)) * 100
        
        for forbidden_state in self.env.forbidden_states:
            Q_values[forbidden_state] = np.full(len(self.env.actions), -200)
            
        for treasure_location in self.env.treasure_locations:
            Q_values[treasure_location] = np.zeros(len(self.env.actions))
            
        print(np.round(np.max(Q_values, axis=2),1))
        return Q_values
    
    @staticmethod
    def weights_gen(n):
        w0 = 0
        while w0 <= 1.0:
            w1 = 1.0 - w0
            yield np.array([w0, w1])
            w0 += 1 / (n-1)
            
            
    
    def q_learning(self, episodes):
        
        
        
    
        #alpha0 = 0.1 # initial learning rate
        epsillon0 = 0.998
        alpha = 0.1
        gamma = 1

        self.stats_dict = {}
        for weights in self.weights_gen(101):
            self.Q_values = self.initialise_q_values()
            stats = []
            for i in range(episodes):
                state = self.env.reset()
                rs = 0
    
                #alpha = max(alpha0 - i / episodes, 0.001) # decay learning rate
                #epsillon = max(epsillon0 - i / episodes, 0.05) # decay epsilon
                epsillon = epsillon0 ** i
    
                while True:
                    action = self.epsillon_greedy_policy(state, epsillon)
                    #print(action)
                    next_state, rewards, done = self.env.step(action)
                    reward = self.scalarise(rewards, weights)
                    rs += reward
                    self.Q_values[(*state, action)] += alpha * (reward +  gamma *
                                                np.max(self.Q_values[next_state]) - 
                                                self.Q_values[(*state, action)])
                    if done:
                        break
                    state = next_state
                stats.append([i, rs])
            key = tuple(np.round(weights, 4))
            self.stats_dict[key] = [np.array(stats), self.Q_values.copy()]
            #self.plot_learning_curve(self.stats_dict[key][0], key)
        
        with open('dst_results.pkl', 'wb') as f:
            pickle.dump(self.stats_dict, f)
        
    def plot_learning_curve(self, stats, key):
        """
        Plot the rewards per episode collected during training
        """
        fig, ax = plt.subplots()
        ax.plot(stats[:,0], stats[:,1])
        ax.set_xlabel('episode')
        ax.set_ylabel('reward per episode')
        ax.set_title(f'time, treasure weighting: {key}')
        plt.show()
 
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
        
        if WEIGHT_IN:
            self.input_size = 4 # state, weights
        else:
            self.input_size = 2 # state
        
    def epsilon_greedy_policy(self, state, weights, epsilon):
        """
        Select greedy action from model output based on current state with 
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            if WEIGHT_IN:
                Q_values = self.model.predict(
                    np.concatenate((state, weights))[np.newaxis])
            else:
                Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values)
        
    def play_one_step(self, state, weights, epsilon):
        """
        Play one action using the DQN and store S A R S' in replay buffer.
        """
        state = np.array(state)
        action = self.epsilon_greedy_policy(state, weights, epsilon)
        
        next_state, rewards, done = self.env.step(action)
        
        next_state = np.array(next_state)
        reward = self.scalarise(rewards, weights)

        self.replay_memory.append((state, action, reward, next_state, done, 
                                   weights))
        return next_state, reward, done
    
    @staticmethod
    def scalarise(rewards, weights):
        rewards = np.array(rewards)
        return np.dot(rewards, weights) # Inner Product
    
    @staticmethod
    def sample_weights(n_obj):
        weights = np.random.rand(n_obj)
        return weights / weights.sum() # Normalise (sum to 1)
    
    def sample_experiences(self):
        """
        Sample a batch from the replay buffer.
        """
        indices = np.random.randint(len(self.replay_memory), 
                                    size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones, weightss = [
                np.array([experience[field_index] for experience in batch])
                for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss
    
    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones, weightss = experiences
        
        # Compute target Q values from 'next_states'
        if WEIGHT_IN:
            next_Q_values = self.model.predict(np.concatenate((next_states, 
                                                               weightss), axis=1))
        else:
            next_Q_values = self.model.predict(next_states)
            
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                       (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector
        
        # Mask to only consider action taken
        mask = tf.one_hot(actions, 4) # 4 actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            if WEIGHT_IN:
                all_Q_values = self.model(np.concatenate((states, 
                                                          weightss), axis=1))
            else:
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

        self.model = keras.Sequential([
            keras.layers.Dense(128, input_shape=(self.input_size,), 
                               activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(4)
            ])

        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss_fn = keras.losses.mean_squared_error
        
    def train_model(self, episodes):
        """
        Train the network over a range of episodes.
        """
        self.build_model()
        
        self.rewards = [] 
        best_reward = -1000
        n_rewards = 10
        reward_list = deque([best_reward for _ in range(n_rewards)], maxlen=n_rewards)
        self.w = [] # TODO remove
        for episode in range(episodes):
            obs = self.env.reset()
            obs = np.array(obs)
            
            weights = self.sample_weights(2) # 2 objectives
            
            self.w.append(weights)
            
            episode_reward = 0
            while True:
                eps = max(self.eps0 - episode / episodes, 0.05) # decay epsilon
                obs, reward, done = self.play_one_step(obs, weights, eps)
                obs = np.array(obs)
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
        #self.model.save(MODEL_PATH)
        
    def play_episode(self, weights):
    
        state = self.env.reset()
        state = np.array(state)
        r = [0,0]
        while True:
            if WEIGHT_IN:
                Q_values = self.model.predict(
                    np.concatenate((state, weights))[np.newaxis])
            else:
                Q_values = self.model.predict(state[np.newaxis])
            action = np.argmax(Q_values)
            state, rewards, done = self.env.step(action)
            
            r[0] += rewards[0]
            r[1] += rewards[1]
            if done:
                print(f'Finished {weights}')
                print(f'{r}\n')
                break
        return r
    
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
        
    
if __name__ == '__main__':
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    
    dst_env = DeepSeaTreasureEnvironment()
    #ag = Agent(dst_env)
    #ag.q_learning(4000)
    ag = DQNAgent(dst_env)
    ag.train_model(3000)
    ag.plot_learning_curve()
    print(ag.play_episode(np.array([0.69841513, 0.30158487])))
    #ag.plot_learning_curve()