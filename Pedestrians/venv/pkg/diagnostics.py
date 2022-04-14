# For collecting diagnostic information about runs
import seaborn as sns
import matplotlib.pyplot as plt
from .general import *
from .dqn import *

sns.set_theme()


def write_training_details(gameconfig, nn_config, fn):
    """
    Write details of a training scenario to a log file
    Include gameconfig, nn_config
    """
    f = open(fn, "w")

    # Write gameconfig
    exclude_gameconfig = ["posconfig", "pos_dict_initial"]
    gameconfig_dict = {
        k: v for k, v in gameconfig.__dict__.items() if k not in exclude_gameconfig
    }
    f.write("gameconfig = GameConfig(\n")
    for key in gameconfig_dict:
        f.write(f"\t{key}: {gameconfig_dict[key]},\n")
    f.write("}")

    f.write("\n")

    # Write nn_config
    nn_config_dict = {k: v for k, v in nn_config.__dict__.items()}
    f.write("nn_config = NNConfig(\n")
    for key in nn_config_dict:
        if key == "mode":
            f.write(f'\t{key} = "{nn_config_dict[key]}",\n')
        else:
            f.write(f"\t{key} = {nn_config_dict[key]},\n")
    f.write(")")


def plot_scores(scores, fn):
    """
    Plot and save fig of the scores (cumulative rewards) of the model agents over a number of episodes
    """
    # Assert all lists are same length
    assert len({len(agent_scores) for agent_scores in scores}) == 1

    # Plot
    episodes = np.arange(1, len(scores[0]) + 1)
    fig = plt.figure()
    for agent, agent_scores in enumerate(scores):
        plt.plot(episodes, agent_scores, label=f"Agent {agent}")
    plt.xlabel("Game")
    plt.ylabel("Cumulative Reward")
    plt.title("Agent Scores during Training")
    plt.legend()
    fig.savefig(fn)


def plot_single_attribute(data, fn, xlabel, ylabel, title):
    """
    Plot a single attribute (e.g. model learning rate) over the course of training
    """
    data_len = len(data)
    x = np.arange(data_len)
    fig = plt.figure()
    plt.plot(x, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(fn)


def mock_game_cnn(cnn, mock_env):
    """Plays a model in a mock_env for one game to see how it fares"""
    # Reset Board
    stop_list = [False for _ in range(mock_env.num_agents)]  # stop recording rewards
    (
        [agent_1, agent_2],
        [target_1, target_2],
        reward_list,
        done_list,
        collided_list,
        reached_list,
        breached_list,
        done,
    ) = mock_env.reset()
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])
    cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])  # padded memory

    # Diagnostics
    game_rewards = [[] for _ in range(mock_env.num_agents)]

    # Play Game
    for move in range(0, mock_env.max_episode_length):

        # Get CNN Actions
        action_list = cnn.act(game=-1, done_list=done_list)

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
        ) = mock_env.step(action_list)

        # Record States
        cnn.update_pos_buffers([agent_1, agent_2], [target_1, target_2])

        # Update Experiences and Diagnostics
        for agent in range(mock_env.num_agents):
            if stop_list[agent]:
                # Don't record rewards after agent is done
                continue
            game_rewards[agent].append(reward_list[agent])

        # Stop list is done list lagged by 1
        stop_list = np.copy(done_list)
        if done:
            break
    return game_rewards


def main():
    gameconfig = GameConfig(
        env_size=256,
        config=1,
        speed=4,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5,
        episodes=100,
    )
    nn_config = NNConfig(
        mode="training",
        mem_max_size=1000,
        minibatch_size=32,
        frac_random=0.1,
        final_epsilon=0.01,
        min_epsilon=0.01,
    )
    write_training_details(gameconfig, nn_config, "hi.txt")
    print(f"Finished {__file__}")


if __name__ == "__main__":
    main()
