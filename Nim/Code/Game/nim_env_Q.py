import random
import numpy as np
from nim_programmed_players import *

class nim_env_Q():
    def __init__(self,i,n,player_1,player_2,first_player = 1,pos_reward=1,neg_reward=-10):
        self.i = i
        self.n = n
        self.tot = 0
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self.player_1 = player_1
        self.player_2 = player_2
        self.done=False
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves
        self.action_space_n = len(self.action_space)
        if not (first_player == 1 or first_player == 2 or first_player == "random"):
            raise ValueError("first player must be 1 or 2 or \"random\"")
        if first_player == 1:
            self.current_player = self.player_1
        elif first_player == 2:
            self.current_player = self.player_2
        elif first_player=="random":
            if random.random() >= 0.5:
                self.current_player = self.player_1
            else:
                self.current_player = self.player_2
        self.state = self.update_state()
    def reset(self):
        self.done=0
        self.tot=0
        return self.update_state()
    def update_state(self):
        self.state = {
            "i":self.i,
            "n":self.n,
            "tot":self.tot,
            "current_player":self.current_player       
            }
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        self.tot += action
        if self.tot<=self.n:
            reward = self.pos_reward
            self.done=False
            #print(self.update_state(), reward_1, reward_2, self.done)
        else:
            # reward_1 = self.pos_reward if self.current_player == self.player_2 else self.neg_reward
            # reward_2 = self.pos_reward if self.current_player == self.player_1 else self.neg_reward
            reward = self.neg_reward
            self.done=True
        if self.current_player == self.player_1:
            self.current_player = self.player_2
        else:
            self.current_player = self.player_1
        return self.update_state(), reward, self.done