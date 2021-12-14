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
    return t%(i+1)

def random_play(i):
    """
    Return random move for a given action set
    """
    return random.randint(1,i)

####################################### main() ####################################

def main():
    random.seed(0)
    print(random_play(10))

if __name__ == "__main__":
    main()
