import os
import datetime
import sys
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
#
# To run from VSCode: comment out lines for sys args and lines setting
# configs using sys args

# Miscellaneous
#random.seed(3)

# Run Script from Colab
args = sys.argv
script,i,n,games,start_player,mode,alpha,gamma,frac_random,final_epsilon,min_epsilon,mem_max_size,reward_mode,skill = args

# Override Parameters

# Override Configs
gameconfig = GameConfig(
        i=3,
        n=20,
        games=2000,
        start_player=0
    )
qconfig = QConfig(
        mode = "training",
        alpha = 0.6,
        gamma = 0.8,
        frac_random = 0.1,
        final_epsilon = 0.001,
        min_epsilon = 0,
        mem_max_size = 1000,
        reward_mode = 0
    )

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

## Create Environment
gameconfig = GameConfig(
        i=int(i),
        n=int(n),
        games=int(games),
        start_player=int(start_player)
    )
env = NimEnv(gameconfig)

## Q Setup
qconfig = QConfig(
        mode = mode,
        alpha = float(alpha),
        gamma = float(gamma),
        frac_random = float(frac_random),
        final_epsilon = float(final_epsilon),
        min_epsilon = float(min_epsilon),
        mem_max_size = int(mem_max_size),
        reward_mode = int(reward_mode)
    )
q = Q(gameconfig, qconfig)
if load_model:
    q.load_q(load_q_fn)
else:
    q.create_q()

# Player Setup
player_0 = q
player_1 = q
players = [player_0, player_1]

### Diagnostics
move_total = 0
wins = np.zeros(gameconfig.games) # indices of winners
optimalities = np.zeros(gameconfig.games) # optimality of q-table after each game

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    i, t, turn, reward_list, done = env.reset()
    prev_action = 0
    prev_turn = turn
    q.update_state_buffer(i,t,turn)

    # Play Game
    while not done:
        
        # Get Action
        action = players[turn].act(i,t,game)

        # Take Action
        i, t, turn, reward_list, done = env.step(action)

        # Record States
        q.update_state_buffer(i,t,turn)

        # Update Experiences
        if done:
            q.update_game_buffer_3(prev_action, reward_list[1-prev_turn], done)
            q.update_game_buffer_2(action, reward_list[prev_turn], done)
        else:
            q.update_game_buffer_3(prev_action, reward_list[prev_turn], done)

        # Update total moves
        move_total+=1

        # Store previous
        prev_action = action
        prev_turn = turn
    # Train
    q.train(move_total, reward_list)

    ### Diagnostics
    wins[game] = turn
    optimalities[game] = table_optimality(gameconfig.i,q.table)

# Plotting
#cum_avg_plot = cum_avg(wins)



print("Finished")
    
