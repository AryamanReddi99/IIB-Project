import random
import numpy as np

class nim_env_DQNvDQN():
    def __init__(self,i,n):
        self.i = i
        self.n = n
        self.observation_space_n = self.n + 1 + i
        self.tot = 0 # total
        self.tot_vec = np.zeros(self.observation_space_n) # vectorised total
        self.turn = 1
        self.done=False
        self.state = np.array([self.tot_vec, self.turn])        
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves: e.g. 0,1,2
        self.action_space_n = len(self.action_space)
    def reset(self):
        self.done=0
        self.tot=0
        self.turn = 1
        return self.update_state()
    def update_state(self):
        self.tot_vec = np.zeros(self.n + self.i + 1) # total of game, vectorised e.g. [1 0 0 0 0]
        self.tot_vec[self.tot] = 1
        self.state = [self.tot_vec,self.turn]
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        self.tot += action
        self.turn *= -1
        if self.tot<=self.n:
            reward = 0 # handle rewards outside
            self.done=False
        else:
            reward = 0
            self.done=True
        return self.update_state(), reward, self.done