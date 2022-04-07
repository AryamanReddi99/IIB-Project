# This script is for making the agents move in programmed ways for demonstration purposes
# This script does not use models of any kind

# Add pkg to path
import os
import sys

sep = os.path.sep  # system path seperator
sys.path.append("/mnt/c/Users/Red/Desktop/Coding/Projects/IIB-Project/Pedestrians/venv")
os.chdir(os.path.dirname(__file__).replace(sep, sep))  # change to cwd

# Imports
import time
import pprint
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.dqn import *
from pkg.window import *

# Miscellaneous
pp = pprint.PrettyPrinter(indent=4)

# Data Paths
fn = Path(__file__).stem  # this filename
store_model_fn = (
    f"..{sep}Saved{sep}"
    + fn
    + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M")
    + f"{sep}Model"
)

# Storage Triggers
store_img = False
store_model = False
load_model = False

# Define Configs
screenconfig = ScreenConfig(headless=False, border_size=10)
gameconfig = GameConfig(
    env_size=128,
    config=11,
    speed=4,
    num_agents=2,
    agent_size=16,
    channels=4,
    num_actions=5,
    games=100,
    doom=False,
)
nn_config = NNConfig(
    mode="testing",
    gamma=0.6,
    mem_max_size=1000,
    minibatch_size=32,
    epoch_size=64,
    frac_random=0.1,
    final_epsilon=0.01,
    min_epsilon=0.01,
    learning_rate=0.001,
    tensorboard=False,
    target_model_iter=10,
)

# Create Functional Classes
window = Window(screenconfig, gameconfig)
env = PedEnv(gameconfig)

# Game Parameters
max_game_length = 100000

### Movement
# Programmed movements, on 256x256 grid, 32 agent size, speed 8
# Doom Agents Ending Slide
actions_agent_1 = [4, 4, 4, 4, 1, 1]
actions_agent_2 = [3, 3, 3, 3, 2, 2]
actions_agent_1_loop = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    1,
    1,
]
actions_agent_2_loop = [
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    1,
    1,
    1,
    1,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    2,
    2,
    2,
    2,
]
actions_agent_1.extend(actions_agent_1_loop * 10000)
actions_agent_2.extend(actions_agent_2_loop * 10000)

# Smooth by speed factor
speed_factor = 8 / gameconfig.speed
repeat_actions = [0, 1, 2, 3, 4]  # repeat these actions to smooth
actions_agent_1_smooth = []
actions_agent_2_smooth = []
for el in actions_agent_1:
    if el in repeat_actions:
        actions_agent_1_smooth.extend([el] * int(speed_factor))
        continue
    actions_agent_1_smooth.append(el)
for el in actions_agent_2:
    if el in repeat_actions:
        actions_agent_2_smooth.extend([el] * int(speed_factor))
        continue
    actions_agent_2_smooth.append(el)

# Extend to fill up total game length
actions_agent_1_smooth.extend([0] * (max_game_length - len(actions_agent_1_smooth)))
actions_agent_2_smooth.extend([0] * (max_game_length - len(actions_agent_2_smooth)))
actions = [
    (actions_agent_1_smooth[step], actions_agent_2_smooth[step])
    for step in range(max_game_length)
]

### Diagnostics
move_total = 0

# Begin Training
for game in tqdm(range(gameconfig.games)):
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
    # cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])
    # cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2]) # padded memory

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
        game=game - 1,
        move=0,
    )
    window.display(display_info=display_info)  # display info on pygame screen

    # Diagnostics
    game_rewards = [[] for i in range(gameconfig.num_agents)]  # rewards for agents

    # Screen Capture Pauses
    if game == 0:
        time.sleep(0)
    time.sleep(0.7)

    # Play Game
    for move in range(0, max_game_length):

        # Get CNN Actions
        # action_list = cnn.act(game, done_list)

        # For testing collisions/targets
        # action_list = [4,3]

        # For running graphics
        action_list = actions[move]

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
        # cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])

        # Update Experiences
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record experiences after agent is done
                continue
            # cnn.update_experiences(agent, action_list, reward_list, done_list)

        # Train
        # cnn.train(move_total)

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
            game=game - 1,
            move=move,
        )
        window.display(display_info=display_info)  # display info on pygame screen

        ### Diagnostics
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record rewards after agent is done
                continue
            game_rewards[agent].append(reward_list[agent])

        # Stop list is done list lagged by 1
        stop_list = np.copy(done_list)
        if done or move == max_game_length - 1:
            if not screenconfig.headless:
                time.sleep(1)
            break

        # Update total moves
        move_total += 1

        # Capture pause
        time.sleep(0.05)

        # First run: prep recorder
        if game == 0 and gameconfig.doom and move == 100:
            break

print("Finished")
