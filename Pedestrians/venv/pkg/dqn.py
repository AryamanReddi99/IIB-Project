from collections import deque, namedtuple
from datetime import datetime
from random import randint, uniform

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from .general import *

################################# Main classes ##############################


class CNN:
    """
    Class that executes all construction, training, and testing of a CNN model.
    Methods:
    - create model
    - store experiences in replay buffer
    - sample replay buffer and train
    - save/load model
    - calculate epsilon
    - execute actions
    - preprocess data
    - create target model
    """

    def __init__(self, gameconfig, nn_config):
        ## gameconfig
        self.env_size = gameconfig.env_size
        self.num_agents = gameconfig.num_agents
        self.channels = gameconfig.channels
        self.num_actions = gameconfig.num_actions
        self.games = gameconfig.games

        ## nn_config
        self.mode = nn_config.mode
        self.gamma = nn_config.gamma
        self.mem_max_size = nn_config.mem_max_size
        self.minibatch_size = nn_config.minibatch_size
        self.epoch_size = nn_config.epoch_size
        self.frac_random = nn_config.frac_random
        self.final_epsilon = nn_config.final_epsilon
        self.min_epsilon = nn_config.min_epsilon
        self.learning_rate = nn_config.learning_rate
        self.tensorboard = nn_config.tensorboard
        self.epochs = nn_config.epochs
        self.target_model_iter = nn_config.target_model_iter

        ## Buffers
        # stores last 2 sets of agent positions
        self.agent_pos_buffer = deque(maxlen=2)
        # stores target positions
        self.target_pos_buffer = deque(maxlen=1)
        # stores anti-target positions
        self.anti_target_pos_buffer = deque(maxlen=1)
        # stores last 2 sets of model-readable states
        self.state_buffer = deque(maxlen=2)
        # stores experiences
        self.replay_buffer = deque(maxlen=nn_config.mem_max_size)

        # Tensorboard
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)

        # Metrics
        self.accuracy = []
        self.loss = []

    def save_cnn(self, path):
        self.model.save(path)

    def load_cnn(self, path):
        self.model = load_model(path)
        self.target_model = load_model(path)
        self._update_target_model()

    def create_cnn(self):
        """
        Creates CNN and Target CNN
        """
        self.model = self._create_sequential()
        self.target_model = self._create_sequential()
        self._update_target_model()

    def update_pos_buffers(self, agent_pos, target_pos):
        # Agents
        agent_pos_data = [None for _ in range(self.num_agents)]
        for agent, pos in enumerate(agent_pos):
            agent_pos_data[agent] = float2mat(pos, self.env_size)
        self.agent_pos_buffer.append(agent_pos_data)

        # Targets
        if len(self.target_pos_buffer) == 0:
            target_pos_data = [None for _ in range(self.num_agents)]
            for target, pos in enumerate(target_pos):
                target_pos_data[target] = float2mat(pos, self.env_size)
            self.target_pos_buffer.append(target_pos_data)

        # Anti-Targets
        if len(self.anti_target_pos_buffer) == 0:
            anti_target_pos_data = [None for _ in range(self.num_agents)]
            for anti_target, pos in enumerate(target_pos):
                anti_target_pos_data[anti_target] = float2mat_anti_target(pos, self.env_size)
            self.anti_target_pos_buffer.append(anti_target_pos_data)

        # Update states
        if len(self.agent_pos_buffer) > 1:
            self._update_state_buffer()

    def act(self, game, done_list):
        """
        Looks at positional data in buffer and creates action
        for each agent
        """
        action_list = [None for _ in range(self.num_agents)]
        for agent in range(self.num_agents):
            # done
            if done_list[agent]:
                action = 0  # stop moving
            # explore
            elif uniform(0, 1) < self._get_epsilon(game):
                action = self._action_space_sample()
            # exploit
            else:
                state = self.state_buffer[-1][agent]
                qvals = self.model.predict(state)
                action = np.argmax(qvals)
            action_list[agent] = action
        return action_list

    def train(self, move_total):
        if self.mode == "testing":
            return
        if len(self.replay_buffer) < self.epoch_size:
            return
        if move_total % self.target_model_iter == 0:
            self._update_target_model()
            print(f"Updated Target Network at move {move_total}")
        self.replay_sample = self._get_replay_sample()
        self._experience_replay()

    def update_experiences(self, agent, action_list, reward_list, done_list):
        """
        Add new experience(s) to replay_buffer
        """
        experience = Experience(
            self.state_buffer[-2][agent],
            action_list[agent],
            reward_list[agent],
            self.state_buffer[-1][agent],
            done_list[agent],
        )
        self.replay_buffer.append(experience)

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _create_sequential(self):
        """
        creates sequential cnn
        """
        model = Sequential()

        # Input, Conv 1
        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=(self.env_size, self.env_size, self.channels),
                padding="same",
                activation="relu",
            )
        )

        # Max Pooling 1
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv 2
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))

        # Max Pooling 2
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten
        model.add(Flatten())

        # Dense 1
        model.add(Dense(256, activation="relu"))

        # Dense 2
        model.add(Dense(128, activation="relu"))

        # Dense 2
        # model.add(Dense(64, activation="relu"))

        # output
        model.add(Dense(self.num_actions))

        # compile
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        return model

    def _update_state_buffer(self):
        """
        Uses last 2 sets of agent positions to update model inputs
        """
        states = [None for _ in range(self.num_agents)]  # [None, None]
        for agent in range(self.num_agents):
            # agent needs current pos, target pos, current + previous pos for each opponent, anti-target pos
            agent_input = [None for _ in range(3 + ((self.num_agents - 1) * 2))]
            # agent's current position
            agent_input[0] = self.agent_pos_buffer[-1][agent]
            # agent's target position
            agent_input[1] = self.target_pos_buffer[-1][agent]
            # agent's anti-target position
            agent_input[2] = self.anti_target_pos_buffer[-1][agent]
            # opponent positions
            insert_position = 3
            for other_agent in range(self.num_agents):
                if agent != other_agent:
                    agent_input[insert_position] = self.agent_pos_buffer[-1][
                        other_agent
                    ]  # where they are
                    agent_input[insert_position + 1] = self.agent_pos_buffer[-2][
                        other_agent
                    ]  # where they were
                    insert_position += 2
            agent_input = arr2stack(np.array(agent_input))
            states[agent] = agent_input
        self.state_buffer.append(states)

    def _get_replay_sample(self):
        indices = np.random.choice(
            len(self.replay_buffer), self.epoch_size, replace=False
        )
        states, actions, rewards, new_states, dones = zip(
            *[self.replay_buffer[index] for index in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(new_states),
            np.array(dones),
        )

    def _experience_replay(self):
        """
        Trains a model on q(s,a) of sampled experiences
        """

        # Receive SARSD training data as 5 numpy arrays
        states, actions, rewards, new_states, dones = self.replay_sample

        # Reshape states and new states into batches
        states_batch = np.squeeze(states)
        new_states_batch = np.squeeze(new_states)

        # Predicted state q values
        target_qvals = self.model.predict(states_batch)

        # Predicted new state q values
        new_states_qvals = self.target_model.predict(new_states_batch)

        # Train on each experience
        for i, (state, action, reward, new_state_qvals, done) in enumerate(
            zip(states, actions, rewards, new_states_qvals, dones)
        ):
            if done:
                target_qval = reward
            else:
                target_qval = reward + self.gamma * np.max(new_state_qvals)
            target_qvals[i][action] = target_qval
        self._cnn_fit(states_batch, target_qvals, self.epochs)

    def _cnn_fit(self, x, y, epochs):
        if self.tensorboard:
            history = self.model.fit(
                x=x,
                y=y,
                batch_size=self.minibatch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[self.tensorboard_callback],
            )
        else:
            history = self.model.fit(
                x=x, y=y, batch_size=self.minibatch_size, epochs=epochs, verbose=1
            )
        loss = history.history["loss"][0]
        accuracy = history.history["accuracy"][0]
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def _get_epsilon(self, game):
        """
        Returns epsilon (random move chance) as decaying e^-x with first
        frac_random*games totally random

        game = current game number
        games = number of total games
        frac_random = fraction of games with epsilon = 1
        final_epsilon = epsilon after all games have been played
        min_epsilon = minimum val of epsilon

        Use game=-1 to put it in testing mode
        """
        if (self.mode == "testing") or game == -1:
            self.epsilon = 0
        else:
            self.epsilon = bound(
                self.min_epsilon,
                1,
                self.final_epsilon
                ** (
                    (game - self.frac_random * self.games)
                    / (self.games * (1 - self.frac_random))
                ),
            )
        return self.epsilon

    def _action_space_sample(self):
        """
        Choose a random move
        """
        return randint(0, self.num_actions - 1)


class NNConfig:
    """
    mode:
    "training" -> works as normal
    "testing" -> doesn't explore (epsilon = 0)
    epoch size = number of samples used in one epoch
    """

    def __init__(
        self,
        mode="training",
        gamma=0.6,
        mem_max_size=1000,
        minibatch_size=32,
        epoch_size=64,
        frac_random=0.1,
        final_epsilon=0.01,
        min_epsilon=0.01,
        learning_rate=0.001,
        tensorboard=False,
        epochs=1,
        target_model_iter=10,
    ):

        self.mode = mode
        self.gamma = gamma
        self.mem_max_size = mem_max_size
        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.frac_random = frac_random
        self.final_epsilon = final_epsilon
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.tensorboard = tensorboard
        self.epochs = epochs
        self.target_model_iter = target_model_iter


################################# External Functions/Classes ##############################

# Tuple class which contains details of an experience
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "new_state", "done"]
)

####################################### main() ####################################
def main():
    gameconfig = GameConfig(
        env_size=256,
        config=1,
        speed=4,
        num_agents=2,
        agent_size=8,
        channels=4,
        num_actions=5,
        games=100,
    )
    nn_config = NNConfig(
        mode="training",
        mem_max_size=1000,
        minibatch_size=32,
        frac_random=0.1,
        final_epsilon=0.01,
        min_epsilon=0.01,
    )
    cnn = CNN(gameconfig, nn_config)
    print("Finished")


if __name__ == "__main__":
    main()
