import os
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.plot import *
from pkg.q import *
from pkg.deterministic import *

# In a game of Q vs Q, each intermediate state must be recorded 
# and used for training.

# Miscellaneous
random.seed(3)

# Data Paths
sep = os.path.sep # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep,sep)) # change to cwd
fn = Path(__file__).stem # this filename
store_q_fn = f"..{sep}Saved{sep}" + fn + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}Q"
load_model_fn = "".replace("\\","/")

# Storage Triggers
store_img = True
store_model = False
load_model = False

# Create Environment
gameconfig = GameConfig(
        i=3,
        n=20,
        games=2000,
        start_player=0
    )
env = NimEnv(gameconfig)

# Q Setup
qconfig = QConfig(
        mode = "training",
        alpha = 0.4,
        gamma = 0.6,
        frac_random = 0.1,
        final_epsilon = 0.001,
        min_epsilon = 0,
        mem_max_size = 1000,
        reward_mode = 2
    )
q = Q(gameconfig, qconfig)
if load_model:
    q.load_q(load_q_fn)
else:
    q.create_q()

# Player Setup
# Learner Always First by Convention (Q-Learner/DQN)
player_0 = q
player_1 = q
#player_1 = ScalablePlayer(1)
players = [player_0, player_1]

### Diagnostics
move_total = 0
wins = np.zeros(gameconfig.games) # indices of winners
optimalities = np.zeros(gameconfig.games) # optimality of q-table after each game

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    i, t, turn, reward_list, done = env.reset()
    prev_turn = turn
    player_0.update_state_buffer(i,t,turn)

    # Play Game
    while not done:
        
        # Get Action
        action = players[turn].act(i,t,game)

        # Take Action
        i, t, turn, reward_list, done = env.step(action)

        # Record States
        player_0.update_state_buffer(i,t,turn)

        # Update Experiences
        if players[prev_turn].trainable:
            players[prev_turn].update_game_buffer(action, reward_list[turn], done)

        # Update total moves
        move_total+=1

        # Update prev_turn
        prev_turn = turn
    # Train
    player_0.train(move_total, reward_list)

    ### Diagnostics
    wins[game] = turn
    optimalities[game] = table_optimality(gameconfig.i,player_0.table)

# Plotting
#cum_avg_plot = cum_avg(wins)



print("Finished")
    
