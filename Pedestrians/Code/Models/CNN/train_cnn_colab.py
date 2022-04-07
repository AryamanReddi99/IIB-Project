# For training the model

# Add pkg to path
import os
import sys

sep = os.path.sep  # system path seperator
sys.path.append("/mnt/c/Users/Red/Desktop/Coding/Projects/IIB-Project/Pedestrians/venv")
os.chdir(os.path.dirname(__file__).replace(sep, sep))  # change to cwd

# Imports
import os
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

# Debugging
pp = pprint.PrettyPrinter(indent=4)


# Data Paths
sep = os.path.sep  # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep, sep))  # change to cwd
fn = Path(__file__).stem  # this filename
store_model_fn = (
    f"..{sep}Saved{sep}"
    + fn
    + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M")
    + f"{sep}Model"
)

# Storage Triggers
store_img = False
store_model = True
load_model = False

# Define Configs
args = sys.argv
(
    script,
    env_size,
    config,
    speed,
    num_agents,
    agent_size,
    channels,
    num_actions,
    games,
    doom,
    mode,
    gamma,
    mem_max_size,
    minibatch_size,
    epoch_size,
    frac_random,
    final_epsilon,
    min_epsilon,
    learning_rate,
    tensorboard,
    target_model_iter,
) = args

screenconfig = ScreenConfig(headless=False, border_size=10)

gameconfig = GameConfig(
    env_size=int(env_size),
    config=int(config),
    speed=int(speed),
    num_agents=int(num_agents),
    agent_size=int(agent_size),
    channels=int(channels),
    num_actions=int(num_actions),
    games=int(games),
    doom=int(doom),
)

nn_config = NNConfig(
    mode=mode,
    gamma=float(gamma),
    mem_max_size=int(mem_max_size),
    minibatch_size=int(minibatch_size),
    epoch_size=int(epoch_size),
    frac_random=float(frac_random),
    final_epsilon=float(final_epsilon),
    min_epsilon=float(min_epsilon),
    learning_rate=float(learning_rate),
    tensorboard=int(tensorboard),
    target_model_iter=int(target_model_iter),
)

# Create Functional Classes
window = Window(screenconfig, gameconfig)
env = PedEnv(gameconfig)
cnn = CNN(gameconfig, nn_config)

# Get Model
if load_model:
    cnn.load_cnn(load_model_fn)
else:
    cnn.create_cnn()

# Game Parameters
max_game_length = 50

### Diagnostics
rewards = [[] for _ in range(gameconfig.num_agents)]
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
        game=game,
        move=0,
    )
    window.display(display_info=display_info)  # display info on pygame screen

    # Diagnostics
    game_rewards = [[] for _ in range(gameconfig.num_agents)]  # rewards for agents

    # Play Game
    for move in range(0, max_game_length):

        # Get CNN Actions
        action_list = cnn.act(game, done_list)

        # For testing collisions/targets
        # action_list = [1,2]

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

        # Update Experiences
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record experiences after agent is done
                continue
            cnn.update_experiences(agent, action_list, reward_list, done_list)

        # Train
        cnn.train(move_total)

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
            game=game,
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
                time.sleep(0.2)
            break

        # Update total moves
        move_total += 1

    # Diagnostics
    game_sum_reward = 0
    for agent in range(gameconfig.num_agents):
        rewards[agent].append(game_rewards[agent])
        game_sum_reward += sum(game_rewards[agent])

    # Good Models
    if game_sum_reward > 0 and store_model:
        cnn.model.save(store_model_fn + f"_game_{game}")
        print(f"Model saved at {store_model_fn}")

# Store Model
if store_model:
    cnn.model.save(store_model_fn)
    print(f"Model saved at {store_model_fn}")

print("Finished")
