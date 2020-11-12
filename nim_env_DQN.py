import random
import numpy as np


class nim_env():
    def __init__(self,i,n):
        self.random_player=random_player()
        self.i = i
        self.n = n
        self.tot = 0
        self.player_flag = 1
        self.done=False
        self.state = np.array([self.tot,self.player_flag])
        #self.action_space = np.array([i for i in range(1,i+1)])
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves
        self.action_space_n = len(self.action_space)
    def reset(self):
        self.done=0
        self.tot=0
        self.player_flag=1
        return self.update_state()
    def update_state(self):
        self.state = np.array([self.tot,self.player_flag])
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        # agent's move
        self.tot += action
        self.player_flag=2
        if self.tot<=self.n:
            reward = 0
            self.done=False
        else:
            reward = -1
            self.done=True
            print(self.update_state(), reward, self.done)
            return self.update_state(), reward, self.done
        # opponent's move
        self.tot += self.random_player.play(self.i,self.n,self.tot,self.player_flag)
        self.player_flag=1
        if self.tot<=self.n:
            reward = 0
            self.done=False
        else:
            reward = 1
            self.done=True
        print(self.update_state(), reward, self.done)
        return self.update_state(), reward, self.done