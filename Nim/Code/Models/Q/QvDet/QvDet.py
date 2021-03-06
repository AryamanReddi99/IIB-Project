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

# In a game of Q-Learner vs deterministic, the q-learner must learn from the states
# given back by the deterministic player. Since they aren't a learner, this is
# essentially like playing the environment, thus there is no need to record 
# intermediate states.


# Miscellaneous
#random.seed(0)

# Run Script from Colab
args = sys.argv
script,i,n,games,start_player,mode,alpha,gamma,frac_random,final_epsilon,min_epsilon,mem_max_size,reward_mode,skill = args

# Override Parameters

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
        i=int(i),
        n=int(n),
        games=int(games),
        start_player=int(start_player)
    )
env = NimEnv(gameconfig)

# Q Setup
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

### Player Setup
# Learner Always First by Convention (Q-Learner/DQN)
player_0 = q

# Deterministic Opponent
player_1 = ScalablePlayer(float(skill))
players = [player_0, player_1]

### Diagnostics
move_total = 0
wins = np.zeros(gameconfig.games) # indices of winners
optimalities = np.zeros(gameconfig.games) # optimality of q-table after each game

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    i, t, turn, reward_list, done = env.reset()

    # Possible Det Move
    if turn==1:
        # Get Action
        action = players[turn].act(i,t,game)

        # Take Action
        i, t, turn, reward_list, done = env.step(action)

    # Record States
    player_0.update_state_buffer(i,t,turn)

    # Play Game
    while not done:
        ##### Turn 0
        # Get Action
        action_0 = player_0.act(i,t,game)
        # Take Action
        i, t, turn, reward_list, done = env.step(action_0)

        ##### Turn 1
        if not done:
            # Get Action
            action_1 = player_1.act(i,t,game)
            # Take Action
            i, t, turn, reward_list, done = env.step(action_1)

        # Record States
        player_0.update_state_buffer(i,t,turn)

        # Update Experiences
        player_0.update_game_buffer_2(action_0, reward_list[0], done)

        # Update total moves
        move_total+=1
    # Train
    player_0.train(move_total, reward_list)

    ### Diagnostics
    wins[game] = turn
    optimalities[game] = table_optimality(gameconfig.i,player_0.table)

# Plotting
#cum_avg_plot = cum_avg(wins)



print("Finished")
    
