import random

def bound(low, high, value):
    """
    Bounds value between low, high
    """
    return max(low, min(high, value))

def optimal_play(i,t):
    """
    The optimal move for any game state of single-heap nim
    """
    optimal = t%(i+1)
    if optimal==0:
        return random_play(i,t)
    return optimal

def random_play(i,t):
    """
    Return random legal move for a given action set
    """
    return random.randint(1,min(i,t))

class GameConfig():
    def __init__(self,
                i=3,
                n=20,
                games = 1000,
                ):
        
        # Game Parameters
        self.i = i
        self.n = n
        self.games = games

####################################### main() ####################################

def main():
    random.seed(0)
    print(random_play(10))

if __name__ == "__main__":
    main()
