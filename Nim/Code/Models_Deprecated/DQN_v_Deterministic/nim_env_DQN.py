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
        self.update_state()
        #self.action_space = np.array([i for i in range(1,i+1)])
        self.observation_space_n = self.n + 1 + i
        self.action_space = np.array([i for i in range(i)]) # creates action space of possible moves: e.g. 0,1,2
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
        return self.update_state()
    def update_state(self):
        self.tot_vec = np.zeros(self.n + self.i + 1) # total of game, vectorised e.g. [1 0 0 0 0]
        self.tot_vec[self.tot] = 1
        self.state = [self.tot_vec,self.player_flag]
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        # agent's move
        self.tot += action
        if self.tot<=self.n:
            reward = 0
            self.done=False
            #print(self.update_state(), reward, self.done)
        else:
            reward = -1
            self.done=True
            return self.update_state(), reward, self.done
        self.player_flag*=-1 # switch player
        # opponent's move
        self.tot += self.opponent.play(self.i,self.n,self.tot,self.player_flag)
        self.player_flag=1
        if self.tot<=self.n:
            reward = 0
            self.done=False
        else:
            reward = +1
            self.done=True
        self.player_flag*=-1 # switch player
        #print(self.update_state(), reward, self.done)
        return self.update_state(), reward, self.done