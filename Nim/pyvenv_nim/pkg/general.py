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
    Return the fraction of the q-table's non-parity states that are correct
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

def table_magnitude(table):
    """
    Return the total abs magnitude of the q-values of a table
    """
    return np.sum(np.absolute(table))

def one_hot(n,dim):
    """
    Convert integer to a one-hot vector of dimension dim+1
    """
    vec = np.zeros(dim + 1)
    vec[n] = 1
    return vec

def one_hot_repeat(n, dim):
    """
    Convert integer to dim+1/n-hot vector of dimension dim+1 with 1 repeated every n places
    """
    n_repeat = 0
    vec = np.zeros(dim + 1)
    while n_repeat<=dim:
        vec[n_repeat]=1
        n_repeat+=n
    return vec

def one_hot_repeat_truncate(n, dim, stop):
    """
    Convert integer to dim+1/i-hot vector of dimension dim+1 with 1 repeated every n places, truncated up to stop
    """
    n_repeat = 0
    vec = np.zeros(dim + 1)
    while n_repeat<=stop:
        vec[n_repeat]=1
        n_repeat+=n
    return vec

def one_hot_action(action, dim):
    """
    Convert integer to one-hot action of dimension dim
    """
    vec = np.zeros(dim)
    vec[action]=1
    return vec

# Tuple class which contains details of an experience
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state', 'done'])

class GameConfig():
    def __init__(self,
                i=3,
                n=20,
                games = 1000,
                max_i = 6,
                max_n = 50,
                start_player = 0
                ):
        
        # Game Parameters
        self.i = i
        self.n = n
        self.games = games
        self.max_i = max_i
        self.max_n = max_n

        # Start Player
        # Can start with 0, 1, or 2 (random)
        self.start_player = start_player

####################################### main() ####################################

def main():
    random.seed(0)
    print(random_play(10))

if __name__ == "__main__":
    main()
