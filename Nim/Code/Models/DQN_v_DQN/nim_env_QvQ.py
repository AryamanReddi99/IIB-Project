import random
import numpy as np

class nim_env_QvQ():
    def __init__(self,i,n):
        self.i = i
        self.n = n
        self.tot = 0
        self.turn = 1
        self.done=False
        self.state = np.array([self.tot, self.turn])
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves
        self.action_space_n = len(self.action_space)
    def reset(self):
        self.done = 0
        self.tot = 0
        self.turn = 1
        return self.tot
    def update_state(self):
        self.state = np.array([self.tot,self.turn])
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        self.tot += action
        self.turn*=-1
        if self.tot<=self.n:
            reward = 0
            self.done=False
        else:
            reward = 0
            self.done=True
        return self.update_state(), reward, self.done