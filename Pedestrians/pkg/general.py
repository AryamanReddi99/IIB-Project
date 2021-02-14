import numpy as np
from env import *

def pos2mat(pos,size):
    """
    Converts position in np array into 1-hot size x size matrix representing x-y position
    e.g. [0,0] in 2x2 grid -> [0 0]
                              [1 0]
    """
    try:
        x = int(pos[0])
        y = int(pos[1])
        mat = np.zeros([size,size])
    except:
        raise TypeError("position must be integer")

    try:
        mat[size-1-y,x] = 1
    except:
        raise TypeError("Matrix size invalid")
    return mat

def bound(low, high, value):
    return max(low, min(high, value))

class GameConfig():
    def __init__(self,  config = 0,
                        size = 256,
                        speed = 4,
                        num_agents = 2,
                        agent_pos = 0,
                        target_pos = 0,
                        agent_size = 5):
        
        self.config = config
        self.size
        self.speed = speed
        self.num_agents = num_agents
        self.agent_size = agent_size
        self.PosConfigs = PosConfigs(self.size)
        self.pos_dict = self.PosConfigs.configs[self.config]

def main():
    print(pos2mat(np.array([2,1]),4))

if __name__=="__main__":
    main()