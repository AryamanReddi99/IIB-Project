import os
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.deterministic import *

# Create Environment
gameconfig = GameConfig(
        i=3,
        n=20,
        games=100
    )
env = NimEnv(gameconfig)

# Player Setup
player_1 = ScalablePlayer(1)
player_2 = ScalablePlayer(1)
players = [player_1, player_2]

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    i, t, turn, reward_list, done = env.reset()

    # Play Game
    while not done:
        
        # Get Action
        action = players[turn].act(i,t)

        # Take Action
        i, t, turn, reward_list, done = env.step(action)

        # Update Experiences

        # Train

    
