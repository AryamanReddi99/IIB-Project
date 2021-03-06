# For testing saves model

# Path
import sys
sys.path.append('c:\\Users\\Red\\Google_Drive\\IIB_Project\\Pedestrians\\pyvenv_ped')

# Imports
import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.dqn import *
from pkg.window import *



# Storage Triggers
store_img = True
store_model = True
load_model = True

# Data Paths
sep = os.path.sep # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep,sep)) # change to cwd
fn = Path(__file__).stem # this filename
load_model_fn = "..\\Saved\\Parallel\\train_cnn-30-05-21_18-08\\Model".replace("\\","/")

# Good Examples
# Best parallel crossing:
#..\\Saved\\Parallel\\train_cnn-11-03-21_17-31\\Model
# Good perp crossing:
#..\\Saved\\Perpendicular\\train_cnn-11-03-21_19-58\\Model


# Create Environment
gameconfig = GameConfig(
    env_size=64,
    config=10,
    speed=10,
    num_agents=2,
    agent_size=7,
    channels=4,
    num_actions=5,
    games=100,
    doom=False)
env = PedEnv(gameconfig)

# Create Display
screenconfig = ScreenConfig(
    headless = False,
    border_size=10)
window = Window(screenconfig, gameconfig)

# CNN Setup
nn_config = NNConfig(
    mode="testing",
    gamma=0.6,
    mem_max_size=1000,
    minibatch_size=32,
    frac_random=0.1,
    final_epsilon=0.01,
    min_epsilon=0.01)
cnn = CNN(gameconfig,nn_config)
if load_model:
    cnn.load_cnn(load_model_fn)
else:
    cnn.create_cnn()

# Game Parameters
max_game_length = 50

### Diagnostics
total_rewards = []

# Begin Training
for game in tqdm(range(gameconfig.games)):
    # Reset Board
    stop_list = [False for _ in range(gameconfig.num_agents)] # stop recording experiences
    [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, breached_list, done = env.reset()
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2]) # padded memory

    # Display Data
    display_info = DisplayInfo(
        agent_pos = [float2pygame(agent_1, gameconfig.env_size), float2pygame(agent_2, gameconfig.env_size)],
        target_pos = [float2pygame(target_1, gameconfig.env_size), float2pygame(target_2, gameconfig.env_size)],
        action_list = [0, 0],
        reward_list = reward_list,
        done_list = done_list,
        collided_list = collided_list,
        reached_list = reached_list,
        breached_list = breached_list,
        done = done,
        game = game,
        move = 0)
    window.display(display_info=display_info) # display info on pygame screen

    ### Diagnostics
    total_reward = 0

    # Play Game
    for move in range(1, max_game_length):

        # Get CNN Actions
        action_list = cnn.act(game, done_list)

        # For testing collisions/targets
        #action_list = [cnn._action_space_sample(),cnn._action_space_sample()]

        # Take Actions 
        [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, breached_list, done = env.step(action_list)
        
        # Record States
        cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])

        # Display Data
        display_info = DisplayInfo(
            agent_pos = [float2pygame(agent_1, gameconfig.env_size), float2pygame(agent_2, gameconfig.env_size)],
            target_pos = [float2pygame(target_1, gameconfig.env_size), float2pygame(target_2, gameconfig.env_size)],
            action_list = action_list,
            reward_list = reward_list,
            done_list = done_list,
            collided_list = collided_list,
            reached_list = reached_list,
            breached_list = breached_list,
            done = done,
            game = game,
            move = move)
        window.display(display_info=display_info) # display info on pygame screen

        ### Diagnostics
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record rewards after agent is done
                continue
            total_reward += reward_list[agent]

        # Update stop_list, check if done:
        stop_list = np.copy(done_list)
        if done or move==max_game_length-1:
            total_rewards.append(round(total_reward, 2))
            time.sleep(1)
            break

print("Finished")

