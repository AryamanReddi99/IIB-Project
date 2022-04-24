### For testing a model

## Imports
import os
import pprint
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import pkg
sep = os.path.sep  # system path seperator
sys.path.append("/mnt/c/Users/Red/Desktop/Coding/Projects/IIB-Project/Pedestrians/venv")
os.chdir(os.path.dirname(__file__).replace(sep, sep))  # change to cwd
from pkg.dqn import *
from pkg.env import *
from pkg.general import *
from pkg.window import *

# Pretty Printer
pp = pprint.PrettyPrinter(indent=4)

# Data Paths
load_model_fn = (
    "../Saved/train_cnn-10-04-22_18-26/train_cnn-10-04-22_18-26/Model_game_381"
)
# load_model_fn = "../Saved/Latest"

# Storage Triggers
store_img = False

# Good Examples
# Best parallel crossing:
# ..\\Saved\\Parallel\\train_cnn-11-03-21_17-31\\Model
# Good perp crossing:
# ..\\Saved\\Perpendicular\\train_cnn-11-03-21_19-58\\Model

# Define Configs
screenconfig = ScreenConfig(headless=False, border_size=1)
gameconfig = GameConfig(
    env_size=8,
    config=11,
    speed=1,
    num_agents=2,
    agent_size=1,
    channels=4,
    num_actions=5,
    episodes=100,
    doom=False,
)
nn_config = NNConfig(
    mode="testing",
    gamma=0.9,
    mem_max_size=2000,
    minibatch_size=32,
    epoch_size=32,
    frac_random=0.3,
    final_epsilon=0.01,
    min_epsilon=0.01,
    learning_rate=0.0001,
    tensorboard=False,
    target_model_iter=10,
)

# Create Functional Classes
window = Window(screenconfig, gameconfig)
env = PedEnv(gameconfig)
cnn = CNN(gameconfig, nn_config)

# Get Model
cnn.load_cnn(load_model_fn)

# Episode Parameters
max_episode_length = 50

### Diagnostics
total_rewards = []

# Begin Testing
for episode in tqdm(range(gameconfig.episodes)):
    # Reset Board
    stop_list = [
        False for _ in range(gameconfig.num_agents)
    ]  # stop recording experiences
    (
        [agent_1, agent_2],
        [target_1, target_2],
        reward_list,
        done_list,
        collided_list,
        reached_list,
        breached_list,
        done,
    ) = env.reset()
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])  # padded memory

    # Display Data
    display_info = DisplayInfo(
        agent_pos=[
            float2pygame(agent_1, gameconfig.env_size),
            float2pygame(agent_2, gameconfig.env_size),
        ],
        target_pos=[
            float2pygame(target_1, gameconfig.env_size),
            float2pygame(target_2, gameconfig.env_size),
        ],
        action_list=[0, 0],
        reward_list=reward_list,
        done_list=done_list,
        collided_list=collided_list,
        reached_list=reached_list,
        breached_list=breached_list,
        done=done,
        episode=episode,
        move=0,
    )
    window.display(display_info=display_info)  # display info on pygame screen

    ### Diagnostics
    total_reward = 0

    # Play Episode
    for move in range(1, max_episode_length):

        # Get CNN Actions
        action_list = cnn.act(episode, done_list)

        # For testing collisions/targets
        # action_list = [cnn._action_space_sample(),cnn._action_space_sample()]

        # Take Actions
        (
            [agent_1, agent_2],
            [target_1, target_2],
            reward_list,
            done_list,
            collided_list,
            reached_list,
            breached_list,
            done,
        ) = env.step(action_list)

        # Record States
        cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])

        # Display Data
        display_info = DisplayInfo(
            agent_pos=[
                float2pygame(agent_1, gameconfig.env_size),
                float2pygame(agent_2, gameconfig.env_size),
            ],
            target_pos=[
                float2pygame(target_1, gameconfig.env_size),
                float2pygame(target_2, gameconfig.env_size),
            ],
            action_list=action_list,
            reward_list=reward_list,
            done_list=done_list,
            collided_list=collided_list,
            reached_list=reached_list,
            breached_list=breached_list,
            done=done,
            episode=episode,
            move=move,
        )
        window.display(display_info=display_info)  # display info on pygame screen

        ### Diagnostics
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record rewards after agent is done
                continue
            total_reward += reward_list[agent]

        # Update stop_list, check if done:
        stop_list = np.copy(done_list)
        if done or move == max_episode_length - 1:
            total_rewards.append(round(total_reward, 2))
            time.sleep(1)
            break

print("Finished")
