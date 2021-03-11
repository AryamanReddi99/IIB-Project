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

# Miscellaneous
pp = pprint.PrettyPrinter(indent=4)

# Data Storage Paths
sep = os.path.sep # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep,sep)) # change to cwd
fn = Path(__file__).stem # this filename
store_model_fn = f"..{sep}Saved{sep}" + fn + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}Model"
load_model_fn = "..\\Saved\\train_cnn-11-03-21_19-58\\Model".replace(sep,"/")

# Storage Triggers
store_img = True
store_model = True
load_model = True

# Create Environment
gameconfig = GameConfig(
    env_size=64,
    config=12,
    speed=10,
    num_agents=2,
    agent_size=8,
    channels=4,
    num_actions=5,
    games=100,
    doom=False)
env = PedEnv(gameconfig)

# Create Display
screenconfig = ScreenConfig(
    headless = True,
    border_size=10)
window = Window(screenconfig, gameconfig)

# CNN Setup
nn_config = NNConfig(
    mode="training",
    gamma=0.6,
    mem_max_size=1000,
    minibatch_size=32,
    epoch_size=64,
    frac_random=0.1,
    final_epsilon=0.01,
    min_epsilon=0.01,
    learning_rate = 0.001,
    tensorboard = True,
    target_model_iter = 10)
cnn = CNN(gameconfig,nn_config)
if load_model:
    cnn.load_cnn(load_model_fn)
else:
    cnn.create_cnn()

# Game Parameters
max_game_length = 50

### Diagnostics

# Begin Training
move_total = 0
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

    # Play Game
    for move in range(0, max_game_length):

        ### Diagnostics

        # Get CNN Actions
        action_list = cnn.act(game, done_list)

        # For testing collisions/targets
        #action_list = [4,3]

        # Take Actions
        [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, breached_list, done = env.step(action_list)
        
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

        # Stop list is delayed done list
        stop_list = np.copy(done_list)
        if done:
            if not screenconfig.headless:
                time.sleep(0.2)
            break

        # Update total moves
        move_total+=1
# Store Model
if store_model:
    cnn.model.save(store_model_fn)
    print(f"Model saved at {store_model_fn}")

### Diagnostics


print("Finished")





























