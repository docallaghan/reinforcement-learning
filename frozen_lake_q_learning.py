import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 23

# Grid Setup
GRID_ROWS = 5
GRID_COLS = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
HOLE_STATES = [(1, 0),
               (1, 3),
               (3, 1),
               (4, 2)]

# Rewards
GOAL_REWARD = 1
HOLE_REWARD = -5.0
TRAN_REWARD = -1.0

# Parameters
P_MOVE = 1.0 # Deterministic with P_MOVE = 1.0
ALPHA = 0.5
GAMMA = 0.9
EPS = 0.1

NUM_EPISODES = 10000

class Environment:
    def __init__(self):

        # Grid structure
        self.grid_rows = GRID_ROWS
        self.grid_cols = GRID_COLS
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE

        # Holes
        #self.num_hole_states = 4 # See comment in create_hole_states()
        #self.hole_states = self.create_hole_states()                                      
        self.hole_states = HOLE_STATES

        # Rewards
        self.goal_reward = GOAL_REWARD
        self.hole_reward = HOLE_REWARD
        self.tran_reward = TRAN_REWARD

    def create_hole_states(self):
        # This method can be used to create random hole states
        # (It was written for the first version of the assignment
        # and is not used in this implementation)
        hole_states = []
        for _ in range(self.num_hole_states):
            while True:
                # Random state on grid
                hole_state = (np.random.randint(GRID_ROWS),
                              np.random.randint(GRID_COLS))
                # Make sure it is not already assigned as 
                # a start, goal or hole state
                if ((hole_state not in hole_states) and
                    (hole_state != self.start_state) and
                    (hole_state != self.goal_state)):
                        hole_states.append(hole_state)
                        break
                    
        return hole_states

    def get_move(self, action):
        # P_MOVE is set to 1.0 so this method always returns
        # the input action
        if np.random.rand() < P_MOVE:
            #print("Deterministic move")
            return action
        elif action in [0, 1]:
            #print("Stochastic move")
            return np.random.choice([2, 3])
        elif action in [2, 3]:
            #print("Stochastic move")
            return np.random.choice([0, 1])
        

    def is_end_state(self, state):
        if ((state in self.hole_states) or
            (state == self.goal_state)):
            return True
        else:
            return False

    def get_next_state(self, state, action):
        move = self.get_move(action)
        if move == 0:                
            next_state = (state[0] - 1, state[1])                
        elif move == 1:
            next_state = (state[0] + 1, state[1])
        elif move == 2:
            next_state = (state[0], state[1] - 1)
        else:
            next_state = (state[0], state[1] + 1)
            
        if (next_state[0] >= 0) and (next_state[0] <= self.grid_rows - 1):
            if (next_state[1] >= 0) and (next_state[1] <= self.grid_cols - 1):                    
                    return next_state # if next state legal
        return state # Any move off the grid leaves state unchanged

    def get_reward(self, state):
        if state == self.goal_state:
            return self.goal_reward        
        elif state in self.hole_states:
            return self.hole_reward
        else:
            return self.tran_reward

    def display_grid(self):
        grid = []
        for i in range(self.grid_rows):
            row = []
            for j in range(self.grid_cols):
                row.append('    .')
            grid.append(row)

        grid[self.start_state[0]][self.start_state[1]] = '    A'
        grid[self.goal_state[0]][self.goal_state[1]] = '    G'
        for hole_state in self.hole_states:
            grid[hole_state[0]][hole_state[1]] = '    H'

        grid = pd.DataFrame(grid)
        print('Frozen Lake:')
        print(grid)
        print()

class Agent:
    def __init__(self, state):
        self.state = state
        self.actions = [0, 1, 2, 3] # up, down, left, right
        
        self.alpha = ALPHA # Learning Rate
        self.gamma = GAMMA # Discount
        self.eps = EPS # Epsillon greedy
            
    def q_learning(self, env, episodes):
        self.q_values = np.zeros((GRID_ROWS, GRID_COLS, len(self.actions)))
        stats = []
        for n in range(episodes):
            s = START_STATE
            rs = 0 # To store reward per episode
            while True:
                a = self.get_action(s) # eps greedy selection
                s_ = env.get_next_state(s, a) # next state
                r = env.get_reward(s_) # reward
                q_max = self.get_qmax(s_)
                self.q_values[s[0], s[1], a] += self.alpha * (r + (self.gamma*q_max) - self.q_values[s[0], s[1], a])
                s = s_
                rs += r # Sum rewards
                
                # Check if terminal
                if env.is_end_state(s):
                    self.q_values[s[0], s[1], :] = env.get_reward(s)
                    break
            stats.append([n, rs])
        self.stats = np.array(stats) # Save for plotting
                
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
        action_grid = np.argmax(ag.q_values, axis=2)
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

if __name__=='__main__':
    # Random Seed
    if SEED != None:
        np.random.seed(SEED)
        
    # Instantiate environment
    env = Environment()
    env.display_grid()
    
    # Instantiate Agent
    ag = Agent(state=START_STATE)
    # Run Q Learning algorithm
    ag.q_learning(env, NUM_EPISODES)
    
    ag.show_q_values()
    ag.show_action_grid()

    # Plotting
    ag.plot_learning_curve()
