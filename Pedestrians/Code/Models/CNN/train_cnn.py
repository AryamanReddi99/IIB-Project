### For training the model

## Imports
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import PrettyPrinter
from time import sleep

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import pkg
sep = os.path.sep  # system path seperator
sys.path.append("/mnt/c/Users/Red/Desktop/Coding/Projects/IIB-Project/Pedestrians/venv")
os.chdir(os.path.dirname(__file__).replace(sep, sep))  # change to cwd
from pkg.diagnostics import *
from pkg.dqn import *
from pkg.env import *
from pkg.general import *
from pkg.window import *

# Pretty Printer
pp = PrettyPrinter(indent=4)

# Data Paths
fn = Path(__file__).stem  # this filename
store_latest_model_fn = f"..{sep}Saved{sep}Latest"
store_model_fn = (
    f"..{sep}Saved{sep}"
    + fn
    + datetime.now().strftime("-%d-%m-%y_%H-%M")
    + f"{sep}Model"
)
store_config_fn = store_model_fn + "_config.txt"
store_img_fn = (
    f"..{sep}Saved{sep}" + fn + datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}"
)
load_model_fn = ""

# Storage Triggers
store_img = True
store_config = True
store_model = True
load_model = False

# Define Configs
screenconfig = ScreenConfig(headless=False, border_size=1)
gameconfig = GameConfig(
    env_size=16,
    config=11,
    speed=1,
    num_agents=2,
    agent_size=3,
    channels=5,
    num_actions=5,
    episodes=100,
    max_episode_length=50,
    reward_target=1,
    reward_death=-1,
    reward_move=-0.05,
    doom=False,
)
nn_config = NNConfig(
    mode="training",
    gamma=0.7,
    mem_max_size=2000,
    minibatch_size=32,
    epoch_size=32,
    frac_random=0.3,
    final_epsilon=0.01,
    min_epsilon=0.01,
    learning_rate=0.00001,
    tensorboard=False,
    target_model_iter=100,
)

# Create Functional Classes
window = Window(screenconfig, gameconfig)
env = PedEnv(gameconfig)
mock_env = PedEnv(gameconfig)
cnn = CNN(gameconfig, nn_config)

# Get Model
if load_model:
    cnn.load_cnn(load_model_fn)
else:
    cnn.create_cnn()

### Diagnostics
rewards = [[] for _ in range(gameconfig.num_agents)]  # rewards for each episode
rewards_mock = [
    [] for _ in range(gameconfig.num_agents)
]  # mock environment rewards for each episode
learning_rate = []  # learning rate of cnn over time
best_cumulative_reward = -100  # score required for a model to get saved
move_total = 0

# Begin Training
for game in tqdm(range(gameconfig.episodes)):

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
    game_rewards = [
        [] for _ in range(gameconfig.num_agents)
    ]  # rewards for all agents for one game

    # Play Game
    for move in range(1, gameconfig.max_episode_length + 1):

        # Get CNN Actions
        action_list = cnn.act(game, done_list)

        # For testing collisions/targets
        action_list = [4, 2]

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

        # Update Experiences and Diagnostics
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record experiences or rewards after agent is done
                continue
            cnn.update_experiences(agent, action_list, reward_list, done_list)
            game_rewards[agent].append(reward_list[agent])

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

        # Update total moves
        move_total += 1

        ## Model Diagnostics
        learning_rate.append(K.eval(cnn.model.optimizer.lr))

        # Stop list is done list lagged by 1
        stop_list = np.copy(done_list)
        if done:
            if not screenconfig.headless:
                sleep(0.2)
            break

    ## Diagnostics
    # Mock environment
    game_rewards_mock = mock_game_cnn(cnn, mock_env)

    # Rewards
    for agent in range(gameconfig.num_agents):
        rewards[agent].append(game_rewards[agent])
        rewards_mock[agent].append(game_rewards_mock[agent])

    # Save models where both agents reach targets and new mock reward highscore reached
    game_total_reward_mock = sum(
        [sum(agent_mock_rewards) for agent_mock_rewards in game_rewards_mock]
    )
    if (
        all(env.reached_list)
        and (game_total_reward_mock > best_cumulative_reward)
        and store_model
    ):
        best_cumulative_reward = game_total_reward_mock
        cnn.model.save(store_model_fn + f"_game_{game}")
        print(f"Model saved at {store_model_fn}")

# Store Final Model and Configs
if store_model:
    cnn.model.save(store_model_fn + f"_latest")
    cnn.model.save(store_latest_model_fn)
    write_training_details(gameconfig, nn_config, store_config_fn)
    print(f"Model saved at {store_model_fn}_latest \nand\n{store_latest_model_fn}")

# Diagnostics Post-Processing
total_rewards = [
    [round(sum(rewards[agent][game]), 2) for game in range(gameconfig.episodes)]
    for agent in range(gameconfig.num_agents)
]  # for each agent, get a list of the total reward at the end of each game
total_rewards_mock = [
    [round(sum(rewards_mock[agent][game]), 2) for game in range(gameconfig.episodes)]
    for agent in range(gameconfig.num_agents)
]  # for each agent, get a list of the total mock reward at the end of each game

## Save Diagnostics
# Score Charts
plot_scores(total_rewards, store_img_fn + f"scores.jpg")
plot_scores(total_rewards_mock, store_img_fn + f"scores_mock.jpg")

# Model Learning Rate
plot_single_attribute(
    data=learning_rate,
    fn=store_img_fn + f"learning_rate.jpg",
    xlabel="Move",
    ylabel="CNN Learning Rate",
    title="CNN Learning Rate During Training",
)

print("Finished")
