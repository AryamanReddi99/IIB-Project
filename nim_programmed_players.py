import random

class scalable_player():
    def __init__(self,skill):
        self.skill = skill
        pass
    def play(self,state,opts=[]):
        a = state["n"]
        while a > state["tot"]:
            prev_a = a
            a -= state["i"]+1
        if a == state["tot"]:
            if "exploit" in opts:
                if random.random() <= self.skill:
                    return 1
                else:
                    return random.randint(1,state["i"])
            else:
                return random.randint(1,state["i"])
        optimal_play = prev_a - state["tot"]
        if random.random() <= self.skill:
            return optimal_play
        else:
            return random.randint(1,state["i"])
class random_player():
    def __init__(self):
        pass
    def act(self,state,opts=None):
        int_input = random.randint(1,state["i"])
        return(int_input)
    def update(self,state,action,reward,next_state):
        pass
class keyboard_player():
    def __init__(self):
        pass
    def play(self,state,opts=None):
        int_input = int(input(f"Human, enter a number between 1 - {i}: "))
        return int_input

