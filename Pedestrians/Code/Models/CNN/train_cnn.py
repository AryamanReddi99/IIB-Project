# Path
import sys
sys.path.append('c:\\Users\\Red\\Google_Drive\\IIB_Project\\Pedestrians\\pyvenv_ped')

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

# Miscellaneous
pp = pprint.PrettyPrinter(indent=4)

# Run Script from Colab
args = sys.argv
script,env_size,config,speed,num_agents,agent_size,channels,num_actions,games,doom,mode,gamma,mem_max_size,minibatch_size,epoch_size,frac_random,final_epsilon,min_epsilon,learning_rate,tensorboard,target_model_iter = args

# Override Parameters
# gameconfig = GameConfig(
#     env_size=64,
#     config=11,
#     speed=10,
#     num_agents=2,
#     agent_size=8,
#     channels=4,
#     num_actions=5,
#     games=100,
#     doom=False)

# nn_config = NNConfig(
#     mode="training",
#     gamma=0.6,
#     mem_max_size=1000,
#     minibatch_size=32,
#     epoch_size=64,
#     frac_random=0.1,
#     final_epsilon=0.01,
#     min_epsilon=0.01,
#     learning_rate = 0.001,
#     tensorboard = False,
#     target_model_iter = 10)

# Data Paths
sep = os.path.sep # system path seperator
os.chdir(os.path.dirname(__file__).replace(sep,sep)) # change to cwd
fn = Path(__file__).stem # this filename
store_model_fn = f"..{sep}Saved{sep}" + fn + datetime.datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}Model"

# Storage Triggers
store_img = True
store_model = True
load_model = False

# Create Environment
gameconfig = GameConfig(
    env_size=env_size,
    config=config,
    speed=speed,
    num_agents=num_agents,
    agent_size=agent_size,
    channels=channels,
    num_actions=num_actions,
    games=games,
    doom=doom)
env = PedEnv(gameconfig)

# Create Display
screenconfig = ScreenConfig(
    headless = True,
    border_size=10)
window = Window(screenconfig, gameconfig)

nn_config = NNConfig(
    mode=mode,
    gamma=gamma,
    mem_max_size=mem_max_size,
    minibatch_size=minibatch_size,
    epoch_size=epoch_size,
    frac_random=frac_random,
    final_epsilon=final_epsilon,
    min_epsilon=min_epsilon,
    learning_rate = learning_rate,
    tensorboard = tensorboard,
    target_model_iter = target_model_iter)
cnn = CNN(gameconfig,nn_config)
if load_model:
    cnn.load_cnn(load_model_fn)
else:
    cnn.create_cnn()

# Game Parameters
max_game_length = 50

### Diagnostics
total_rewards = []
move_total = 0

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
    for move in range(0, max_game_length):

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

        ### Diagnostics
        for agent in range(gameconfig.num_agents):
            if stop_list[agent]:
                # Don't record rewards after agent is done
                continue
            total_reward += reward_list[agent]

        # Stop list is done list lagged by 1
        stop_list = np.copy(done_list)
        if done or move==max_game_length-1:
            total_rewards.append(round(total_reward, 2))
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





























