import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from datetime import datetime
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.dqn import *
from pkg.window import *

def main():

    # Data Storage
    os.chdir(os.path.dirname(__file__))
    store_img = True
    store_model = True
    load_model = False
    store_fn = "cnn-" + datetime.now().strftime("%d/%m/%y-%H:%M") 
    load_fn = "cnn-fn"

    # Create Environment
    env_size = 256 # board size
    posconfig = PosConfig(size=env_size)
    gameconfig = GameConfig(
        posconfig=posconfig,
        config=1,
        speed=8,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5)
    env = PedEnv(gameconfig)

    # Create Display
    screenconfig = ScreenConfig(border_size=10)
    window = Window(screenconfig, gameconfig)

    # Create Model
    if load_model:
        model = load_model(load_fn)
    else:
        model = create_model(gameconfig)

    # Training Hyperparameters
    gamma = 0.6
    
    # Training Parameters
    games = 100
    max_game_length = 50
    minibatch_size = 32
    mem_max_size = 100
    replay_memory = np.array([])

    # Run Simulations
    for game in tqdm(range(games)):
        # Reset Board

        # Initial Positions
        [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, done = env.reset()
        agent_1_prev = agent_1
        agent_2_prev = agent_2

        # Display Data
        display_info = {
            "agent_pos": [agent_1, agent_2],
            "target_pos": [target_1, target_2],
            "action_list": [0, 0],
            "reward_list": reward_list,
            "done_list": done_list,
            "collided_list": collided_list,
            "reached_list": reached_list,
            "done": done,
            "game": game,
            "move": 0}

        window.display(display_info=display_info) # display info on pygame screen

        for move in range(1, max_game_length):
            # Epsilon Update
            epsilon = get_epsilon(game=game,frac_random=0.1,final_epsilon=0.01,num_games=games)

            # Reformat Coordinate Data into Matrices
            agent_1_prev_mat = float2mat(agent_1_prev, gameconfig.size)
            agent_2_prev_mat = float2mat(agent_2_prev, gameconfig.size)
            agent_1_mat = float2mat(agent_1, gameconfig.size)
            agent_2_mat = float2mat(agent_2, gameconfig.size)
            target_1_mat = float2mat(target_1, gameconfig.size)
            target_2_mat = float2mat(target_2, gameconfig.size)

            # Agent 1 Move
            agent_1_input = np.array([agent_1_mat,target_1_mat,agent_2_mat,agent_2_prev_mat]).reshape(1,gameconfig.size,gameconfig.size,gameconfig.channels)
            qvals_s_1 = model.predict(agent_1_input)
            if random.uniform(0,1) < epsilon:
                action_1 = env.action_space_sample() # Explore
            else:
                action_1 = np.argmax(qvals_s_1) # Exploit

            # Agent 2 Move
            agent_2_input = np.array([agent_2_mat,target_2_mat,agent_1_mat,agent_1_prev_mat]).reshape(1,gameconfig.size,gameconfig.size,gameconfig.channels)
            qvals_s_2 = model.predict(agent_2_input)
            if random.uniform(0,1) < epsilon:
                action_2 = env.action_space_sample() # Explore
            else:
                action_2 = np.argmax(qvals_s_2) # Exploit  

            # For testing collisions/targets
            #action_1 = 1
            #action_2 = 3 


            # Store Previous States and Take Actions
            agent_1_prev = agent_1
            agent_2_prev = agent_2
            [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, done = env.step([action_1, action_2])

            # Display Data
            display_info = {
                "agent_pos": [agent_1, agent_2],
                "target_pos": [target_1, target_2],
                "action_list": [action_1, action_2],
                "reward_list": reward_list,
                "done_list": done_list,
                "collided_list": collided_list,
                "reached_list": reached_list,
                "done": done,
                "game": game,
                "move": move}

            window.display(display_info=display_info) # display info on pygame screen

            # Training Data

            # Check if Done
            if done:
                time.sleep(1)
                break

if __name__=="__main__":
    main()





























