import random
import numpy as np
from nim_programmed_players import *

class nim_env_DQN():
    def __init__(self,i,n,opponent,first_player = "agent"):
        self.opponent=opponent
        self.i = i
        self.n = n
        self.tot = 0
        if not (first_player == "agent" or first_player == "opponent" or first_player == "random"):
            raise ValueError("first player must be \"agent\" or \"opponent\"")
        self.first_player = first_player
        self.player_flag = 1
        self.done=False
        self.state = np.array([self.tot,self.player_flag])
        #self.action_space = np.array([i for i in range(1,i+1)])
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves
        self.action_space_n = len(self.action_space)
    def reset(self):
        self.done=0
        self.tot=0
        # make random player start
        if self.first_player == "opponent":
            self.tot += self.opponent.play(self.i,self.n,self.tot,self.player_flag) 
        elif self.first_player == "random":
            if random.random > 0.5:
                self.tot += self.opponent.play(self.i,self.n,self.tot,self.player_flag) 
        self.player_flag=1
        return self.tot
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
            #print(self.update_state(), reward, self.done)
        else:
            reward = -10
            self.done=True
            return self.tot, reward, self.done
        # opponent's move
        self.tot += self.opponent.play(self.i,self.n,self.tot,self.player_flag)
        self.player_flag=1
        if self.tot<=self.n:
            reward = 0
            self.done=False
        else:
            reward = 1
            self.done=True
        #print(self.update_state(), reward, self.done)
        return self.tot, reward, self.done