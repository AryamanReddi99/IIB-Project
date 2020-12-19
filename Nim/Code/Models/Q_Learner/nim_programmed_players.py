import random

class scalable_player():
    def __init__(self,skill):
        self.skill = skill
        pass
    def play(self,i,n,tot,player_flag,opts=[]):
        a = n
        while a > tot:
            prev_a = a
            a -= i+1
        if a == tot:
            if "exploit" in opts:
                if random.random() <= self.skill:
                    return 1
                else:
                    return random.randint(1,i)
            else:
                return random.randint(1,i)
        optimal_play = prev_a - tot
        if random.random() <= self.skill:
            return optimal_play
        else:
            return random.randint(1,i)

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
