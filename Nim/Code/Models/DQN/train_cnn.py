import os
import datetime
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.plot import *
from pkg.dqn import *
from pkg.deterministic import *

# In a game of DQN vs DQN each intermediate state must be recorded 
# and used for training.
#
# To run from VSCode: comment out lines for sys args and lines setting
# configs using sys args

# Miscellaneous
random.seed(0)

# Run Script from Colab
# args = sys.argv
# script,i,n,games,start_player,mode,alpha,gamma,frac_random,final_epsilon,min_epsilon,mem_max_size,reward_mode,skill = args

# Override Parameters

# Manual Configs
gameconfig = GameConfig(
        i=3,
        n=20,
        games=10,
        max_i = 3,
        max_n = 20,
        start_player=0,
    )
nn_config = NNConfig(
        mode = "training",
        gamma = 0.6,
        mem_max_size = 1000,
        minibatch_size = 32,
        epoch_size = 64,
        num_filters = gameconfig.max_i+1,
        kernel_regulariser = 0.001,
        kernel_activation = 'relu',
        truncate_i = False,
        frac_random = 0.1,
        final_epsilon = 0.0001,
        min_epsilon = 0,
        learning_rate = 0.001,
        tensorboard = False,
        epochs = 1,
        target_model_iter = 10,
        reward_mode = 0,
        optimal_override = 0,
        test_divisors = [3]
)


# Data Paths
sep = os.path.sep # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep,sep)) # change to cwd
fn = Path(__file__).stem # this filename
store_cnn_fn = f"..{sep}Saved{sep}" + fn + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}Q"
load_cnn_fn = "".replace("\\","/")

# Storage Triggers
store_img = True
store_model = False
load_model = False

## Create Environment
# gameconfig = GameConfig(
#         i=int(i),
#         n=int(n),
#         games=int(games),
#         start_player=int(start_player)
#     )

env = NimEnv(gameconfig)

cnn = CNN(gameconfig, nn_config)
if load_model:
    cnn.load_cnn(load_cnn_fn)
else:
    cnn.create_cnn()

# Player Setup
player_0 = cnn
player_1 = cnn
players = [player_0, player_1]

### Diagnostics
move_total = 0
wins = np.zeros(gameconfig.games) # indices of winners
optimalities = np.zeros(gameconfig.games) # optimality of network after each game

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    i, t, turn, reward_list, done = env.reset()
    prev_action = 0
    prev_turn = turn
    cnn.update_state_buffer(i,t,turn)

    # Play Game
    while not done:
        
        # Get Action
        action = cnn.act(game)

        # Take Action
        i, t, turn, reward_list, done = env.step(action)

        # Record States
        cnn.update_state_buffer(i,t,turn)

        # Update Experiences
        if done:
            cnn.update_game_buffer_3(prev_action, reward_list[1-prev_turn], done)
            cnn.update_game_buffer_2(action, reward_list[prev_turn], done)
        else:
            cnn.update_game_buffer_3(prev_action, reward_list[prev_turn], done)

        # Update total moves
        move_total+=1

        # Store previous
        prev_action = action
        prev_turn = turn

        # Train
        cnn.train(move_total)

    # Update Game Buffer 
    cnn.update_experiences(reward_list)

    ### Diagnostics
    wins[game] = turn
    optimalities[game] = cnn.total_optimality()

print("Finished")
    
