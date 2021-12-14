# deterministic class functions - scalable players, human players
import random

class ScalablePlayer():
    """
    Plays optimal moves with probability $skill, 
    plays randomly with probability 1 - $skill
    """
    def __init__(self,skill):
        self.skill = skill
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

class KeyboardPlayer():
    def __init__(self):
        pass
    def play(self,state,opts=None):
        int_input = int(input(f"Please enter a number between 1 - {i}: "))
        return int_input

class DeterministicGame():
    """
    determinstic single-pile final-pickup game Nim between humans or agents
    """
    def __init__(self,i,lim):
        self.i = i
        self.lim = lim
        self.tot = 0
        self.player_flag = 1

    def get_move(self,player,opts=None):
        # get a move from a player and check validity
        try:    
            int_input = int(player.play(self.i,self.lim,self.tot,self.player_flag))
        except:
            print("\n")
            print("WARNING: Input must be a number.")
            print("\n")
            int_input = self.get_move(self.player_1,opts)
        if int_input > self.i:
            print("\n")
            print(f"WARNING: Input must be between 1 - {i}.")
            print("\n")
            int_input = self.get_move(self.player_1)
        if self.verbose:
            print(f"Player {self.player_flag} played {int_input}")
        return int_input

    def play_turn(self):
        # allow player to play and switch player flag
        if self.player_flag==1:
            num = self.get_move(self.player_1)
            self.player_flag=2
        else:
            num = self.get_move(self.player_2)
            self.player_flag=1
        return num

    def play(self,player_1=keyboard_player(),player_2=keyboard_player(),verbose=False):
        self.tot = 0
        self.verbose=verbose
        self.player_flag=1
        self.player_1=player_1
        self.player_2=player_2
        if self.verbose:
            print(f"First to exceed {self.lim} loses!")
            print("Input 0 to reset game")
            print("Input -1 to end game")
            print("\n")
            while self.tot <= self.lim:
                print(f"Total = {self.tot}\n")    
                num = self.play_turn()
                if num == 0:
                    self.reset()
                    print("\nGame reset")
                elif num==-1:
                    self.player_flag=-1
                    break
                else:
                    self.tot += num
            # after game is over
            if self.player_flag == 2:
                print("\n")
                print("Player 2 wins!")
            elif self.player_flag == 1:
                print("\n")
                print("Player 1 wins!")
            elif self.player_flag == -1:
                print("\n")
                print("Game cancelled")
        else:
            while self.tot <= self.lim:
                num = self.play_turn()
                if num == 0:
                    self.reset()
                elif num==-1:
                    self.player_flag=-1
                    break
                else:
                    self.tot += num
            # after game is over
        return self.player_flag # player flag indicates the winner
    def reset(self):
            # used to reset a game
            self.tot = 0
            self.player_flag = 1

####################################### main() ####################################

