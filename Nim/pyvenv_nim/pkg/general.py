import random
import collections
import numpy as np

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

def consecutive(data, stepsize=1):
    """
    Returns arrays of indices where data is consecutive
    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def table_optimality(i,table):
    """
    Return the % of the q-table's non-parity states that are correct
    """
    # Only evaluate non-parity states
    evaluatable=0
    # Count how many rows have the correct argmax
    correct=0
    for t,row in enumerate(table):
        optimal = t%(i+1)
        if optimal==0:
            continue
        elif (np.argmax(row)+1==optimal) and (np.count_nonzero(row==np.max(row))==1):
            correct+=1
        evaluatable+=1
    return correct/evaluatable

# Tuple class which contains details of an experience
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state', 'done'])

class GameConfig():
    def __init__(self,
                i=3,
                n=20,
                games = 1000,
                start_player = 0
                ):
        
        # Game Parameters
        self.i = i
        self.n = n
        self.games = games

        # Start Player
        # Can start with 0, 1, or 2 (random)
        self.start_player = start_player

####################################### main() ####################################

def main():
    random.seed(0)
    print(random_play(10))

if __name__ == "__main__":
    main()
