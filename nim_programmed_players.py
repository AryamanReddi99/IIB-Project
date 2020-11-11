import random

class random_player():
    def __init__(self):
        pass
    def play(self,i,lim,tot,player_flag,env_move=None):
        int_input = random.randint(1,i)
        return(int_input)


class keyboard_player():
    def __init__(self):
        pass
    def play(self,i,lim,tot,player_flag,env_move):
        int_input = int(input(f"Player {player_flag} enter a number between 1 - {i}: "))
        return int_input

class dqn_player_interface():
    def __init__(self):
        pass
    def play(self,i,lim,tot,player_flag,env_move):
        return env_move
