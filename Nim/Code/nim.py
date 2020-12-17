import numpy as np
import matplotlib.pyplot as plt

from nim_programmed_players import *
from scipy.stats import norm

class game():
    """
    class which plays single-pile final-pickup Nim between humans or agents
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

a = scalable_player(1)
b = scalable_player(1)


game_1 = game(3,24)
winners = []
for i in range(1000):
    winners.append(game_1.play(a,b))
print(winners)





# num_wins_1_list = []
# for _ in range(big_iter):
#     num_wins_1 = 0
#     for _ in range(small_iter):
#         game_result = game_1.play(a,b)
#         if game_result == 1:
#             num_wins_1 += 1 
#     num_wins_1_list.append(num_wins_1)

# num_wins_2_list = [small_iter-i for i in num_wins_1_list]

# mu1, std1 = norm.fit(num_wins_1_list)
# mu2, std2 = norm.fit(num_wins_2_list)

# x=np.linspace(20,80,100)
# p1 = norm.pdf(x, mu1, std1)
# p2 = norm.pdf(x, mu2, std2)
# plt.plot(x,p1,color="purple",linewidth=2)
# plt.plot(x,p2,color="orange", linewidth=2)



# print(mu1,std1)
# print(mu2,std2)
# plt.hist(num_wins_1_list,bins=np.arange(20,81),density=True,color="blue")
# plt.hist(num_wins_2_list,bins=np.arange(20,81),density=True,color="red")
# plt.show()