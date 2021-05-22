from .general import *

class NimEnv():
    def __init__(self, gameconfig):

        # Game Params
        self.i = gameconfig.i
        self.n = gameconfig.n

        # Env Params
        self.done = 0
        self.turn = 0
        self.reward_lose = -3
        self.reward_win = +3

    def reset(self):
        # Starting parameters
        self.done = 0
        self.turn = 0
        self.t = self.n

        # Rewards
        self.reward_list = [0 for _ in range(2)]

        return (self.i, self.t, self.turn, self.reward_list, self.done)

    def action_space_sample(self):
        """
        Return random action
        """
        return random_play(self.i,self.t)

    def step(self, action):
        self.t -= action
        if self.t == 0:
            # Turn player won
            self.reward_list[self.turn] = self.reward_win
            self.reward_list[1-self.turn] = self.reward_lose
            self.done = 1
        elif self.t < 0:
            # Turn player lost
            self.reward_list[self.turn] = self.reward_lose
            self.reward_list[1-self.turn] = self.reward_win
            self.done = 1
        else:
            self.turn = 1 - self.turn
        
        return (self.i, self.t, self.turn, self.reward_list, self.done)

####################################### main() ####################################

def main():
    gameconfig = GameConfig(
        i=3,
        n=20
    )
    env = NimEnv(gameconfig)
    print("Finished")

if __name__=="__main__":
    main()
