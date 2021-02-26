import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from datetime import datetime
from pathlib import Path; print(Path("/path/to/some/file.txt").stem)
from tqdm import tqdm
from pkg.general import *
from pkg.env import *
from pkg.dqn import *
from pkg.window import *

def main():

    # Data Storage Paths
    sep = os.path.sep # system path seperator
    os.chdir(os.path.dirname(__file__)) # change to cwd
    fn = Path(__file__).stem # this filename
    store_model_fn = f"..{sep}Saved{sep}" + fn + datetime.now().strftime("-%d-%m-%y_%H-%M") + f"{sep}Model"
    load_model_fn = "Pedestrians\Code\Models\Saved\ped_cnn-19-02-21_03-37\Model".replace(sep,sep)

    # Storage Triggers
    store_img = True
    store_model = True
    load_model = False

    # Create Environment
    env_size=256
    gameconfig = GameConfig(
        env_size=env_size,
        config=1,
        speed=10,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5,
        doom=True)
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
    training_mode = False
    games = 10
    max_game_length = 50
    minibatch_size = 8
    mem_max_size = 1000
    delay_training = minibatch_size
    assert(delay_training >= minibatch_size)
    buffer = ReplayMemory(mem_max_size)

    # Begin Training
    for game in tqdm(range(games)):
        # Reset Board

        # Initial Positions
        [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, done = env.reset()
        [agent_1_prev, agent_2_prev] = [agent_1, agent_2]

        # Reformat Initial Coordinate Data into Matrices
        agent_1_prev_mat = float2mat(agent_1_prev, env_size)
        agent_2_prev_mat = float2mat(agent_2_prev, env_size)
        agent_1_mat = float2mat(agent_1, env_size)
        agent_2_mat = float2mat(agent_2, env_size)
        target_1_mat = float2mat(target_1, env_size)
        target_2_mat = float2mat(target_2, env_size)

        # Reformat Initial Data into Model-Friendly Form
        agent_1_input = np.array([agent_1_mat,target_1_mat,agent_2_mat,agent_2_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)
        agent_2_input = np.array([agent_2_mat,target_2_mat,agent_1_mat,agent_1_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)

        # Display Data
        display_info = DisplayInfo(
            agent_pos = [pos2pygame(agent_1, env_size), pos2pygame(agent_2, env_size)],
            target_pos = [pos2pygame(target_1, env_size), pos2pygame(target_2, env_size)],
            action_list = [0, 0],
            reward_list = reward_list,
            done_list = done_list,
            collided_list = collided_list,
            reached_list = reached_list,
            done = done,
            game = game,
            move = 0)
        window.display(display_info=display_info) # display info on pygame screen

        # Play Game
        for move in range(1, max_game_length):
            # Epsilon Update
            epsilon = get_epsilon(game=game,frac_random=0.1,final_epsilon=0.01,min_epsilon=0.02,num_games=games)

            # Agent 1 Move
            agent_1_input = np.array([agent_1_mat,target_1_mat,agent_2_mat,agent_2_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)
            if random.uniform(0,1) < epsilon:
                action_1 = env.action_space_sample() # Explore
            else:
                qvals_1 = model.predict(agent_1_input)
                action_1 = np.argmax(qvals_1) # Exploit

            # Agent 2 Move
            agent_2_input = np.array([agent_2_mat,target_2_mat,agent_1_mat,agent_1_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)
            if random.uniform(0,1) < epsilon:
                action_2 = env.action_space_sample() # Explore
            else:
                qvals_2 = model.predict(agent_2_input)
                action_2 = np.argmax(qvals_2) # Exploit  

            # For testing collisions/targets
            #action_1 = 1
            #action_2 = 3 

            action_list = [action_1, action_2]

            # If done, don't move
            for agent, done_state in enumerate(done_list):
                if done_state:
                    action_list[agent] = 0

            # Store Previous States and Take Actions
            [agent_1_prev, agent_2_prev] = [agent_1, agent_2]
            [agent_1, agent_2], [target_1, target_2], reward_list, done_list, collided_list, reached_list, done = env.step(action_list)

            # Reformat Coordinate Data into Matrices
            agent_1_prev_mat = float2mat(agent_1_prev, env_size)
            agent_2_prev_mat = float2mat(agent_2_prev, env_size)
            agent_1_mat = float2mat(agent_1, env_size)
            agent_2_mat = float2mat(agent_2, env_size)
            target_1_mat = float2mat(target_1, env_size)
            target_2_mat = float2mat(target_2, env_size)

            # Reformat data into model-friendly form
            agent_1_prev_input = agent_1_input
            agent_2_prev_input = agent_2_input
            agent_1_input = np.array([agent_1_mat,target_1_mat,agent_2_mat,agent_2_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)
            agent_2_input = np.array([agent_2_mat,target_2_mat,agent_1_mat,agent_1_prev_mat]).reshape(1,env_size,env_size,gameconfig.channels)

            # Training Data
            if not done_list[0]:
                experience_1 = Experience(agent_1_prev_input, action_1, reward_list[0], agent_1_input, done_list[0])
                buffer.append(experience_1)
            if not done_list[1]:
                experience_2 = Experience(agent_2_prev_input, action_2, reward_list[1], agent_2_input, done_list[1])
                buffer.append(experience_2)

            # Train
            if training_mode:
                if buffer.len() >= delay_training:
                    replay_sample = buffer.replay_sample(minibatch_size)
                    model = experience_replay(model, replay_sample, gamma)

            # Display Data
            display_info = DisplayInfo(
                agent_pos = [pos2pygame(agent_1, env_size), pos2pygame(agent_2, env_size)],
                target_pos = [pos2pygame(target_1, env_size), pos2pygame(target_2, env_size)],
                action_list = [action_1, action_2],
                reward_list = reward_list,
                done_list = done_list,
                collided_list = collided_list,
                reached_list = reached_list,
                done = done,
                game = game,
                move = move)

            window.display(display_info=display_info) # display info on pygame screen

            # Check if Done
            if done:
                time.sleep(1)
                break

    # Store Model
    if store_model:
        model.save(store_model_fn)

    print("Finished")

if __name__=="__main__":
    main()





























