# deterministic class functions - scalable players, human players
import random
from .general import *

class ScalablePlayer():
    """
    Plays optimal moves with probability $skill, 
    plays randomly with probability 1 - $skill
    """
    def __init__(self,skill):
        self.skill = skill
        self.trainable = False

    def act(self, i, t, game):
        if random.random() <= self.skill:
            return optimal_play(i,t)
        else:
            return random_play(i,t)

class KeyboardPlayer():
    def __init__(self):
        pass
    def act(self,state):
        int_input = int(input(f"Please enter a number between 1 - {i}: "))
        return int_input

####################################### main() ####################################

