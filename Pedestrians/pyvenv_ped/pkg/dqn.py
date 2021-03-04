import collections
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Conv2D, MaxPooling2D
from .general import *

class CNN():
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
        self.frac_random = nn_config.frac_random
        self.final_epsilon = nn_config.final_epsilon
        self.min_epsilon = nn_config.min_epsilon

        ## Buffers
        # experiences
        self.replay_buffer = ReplayMemory(capacity=nn_config.mem_max_size)
        # stores last 2 sets of agent positions
        self.agent_pos_buffer = collections.deque(maxlen=2)
        # stores last 2 sets of target positions
        self.target_pos_buffer = collections.deque(maxlen=1)
        # states
        self.state_buffer = collections.deque(maxlen=2)

    def save_cnn(self, path):
        self.model.save(path)

    def load_cnn(self, path):
        self.model = load_model(path)

    def create_cnn(self):
        """
        Creates Sequential CNN
        """
        model = Sequential()

        # Input, Conv 1
        model.add(Conv2D(32, (3,3), input_shape=(self.env_size,self.env_size,self.channels), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # conv 2
        model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # conv 3
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        # model.add(BatchNormalization())

        # Flatten
        model.add(Flatten())

        # Dense 1
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # Dense 2
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # output
        model.add(Dense(self.num_actions))

        # compile
        model.compile(
            loss='mean_squared_error',
            optimizer='Adam',
            metrics=['accuracy'])

        self.model = model

    def update_pos_buffers(self, agent_pos, target_pos):
        # Agents
        agent_pos_data = [None for _ in range(self.num_agents)]
        for agent, pos in enumerate(agent_pos):
            agent_pos_data[agent] = self._matrixify(pos)
        self.agent_pos_buffer.append(agent_pos_data)

        # Targets
        if len(self.target_pos_buffer) == 0:
            target_pos_data = [None for _ in range(self.num_agents)]
            for target, pos in enumerate(target_pos):
                target_pos_data[target] = self._matrixify(pos)
            self.target_pos_buffer.append(target_pos_data)

        # Update inputs 
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
                action = 0 # stop moving
            # explore
            elif random.uniform(0,1) < self._get_epsilon(game):
                action = self._action_space_sample()
            # exploit
            else:
                state = self.state_buffer[-1][agent]
                qvals = self.model.predict(state)
                action = np.argmax(qvals)
            action_list[agent] = action
        return action_list

    def train(self):
        if self.mode == "testing":
            return
        if self.replay_buffer.len() < self.minibatch_size:
            return
        self.replay_sample = self.replay_buffer.replay_sample(self.minibatch_size)
        self._experience_replay()

    def update_experiences(self, agent, action_list, reward_list, done_list):
        """
        Add new experience(s) to replay_buffer
        """
        experience = Experience(self.state_buffer[-2][agent], action_list[agent], reward_list[agent], self.state_buffer[-1][agent], done_list[agent])
        self.replay_buffer.append(experience)

    def _update_state_buffer(self):
        """
        Uses last 2 sets of agent positions to update model inputs
        """
        states = [None for _ in range(self.num_agents)] # [None, None]
        for agent in range(self.num_agents):
            # agent needs current pos, target pos, current + previous pos for each opponent
            agent_input = [None for _ in range(2 + ((self.num_agents - 1)*2))]
            # agent's current position
            agent_input[0] = self.agent_pos_buffer[-1][agent]
            # agent's target position
            agent_input[1] = self.target_pos_buffer[-1][agent]
            # opponent positions
            insert_position = 2
            for other_agent in range(self.num_agents):
                if agent != other_agent:
                    agent_input[insert_position] = self.agent_pos_buffer[-1][other_agent] # where they are
                    agent_input[insert_position + 1] = self.agent_pos_buffer[-2][other_agent] # where they were
                    insert_position += 2
            agent_input = arr2stack(np.array(agent_input))
            states[agent] = agent_input
        self.state_buffer.append(states)

    def _experience_replay(self):
        """
        Trains a model on q(s,a) of sampled experiences
        """

        # Recieve SARSD training data as 5 numpy arrays
        states, actions, rewards, new_states, dones = self.replay_sample

        # Reshape states and new states into batches
        states_batch = np.squeeze(states)
        new_states_batch = np.squeeze(new_states)

        # Predicted state q values
        target_qvals = self.model.predict(states_batch)

        # Predicted new state q values
        new_states_qvals = self.model.predict(new_states_batch)

        # Train on each experience
        for i, (state,action,reward,new_state_qvals,done) in enumerate(zip(states,actions,rewards,new_states_qvals,dones)): 
            if done:
                target_qval = reward
            else:
                target_qval = reward + self.gamma * np.max(new_state_qvals)
            target_qvals[i][action] = target_qval
        self.model.fit(states_batch, target_qvals, batch_size=self.minibatch_size, epochs=1, verbose=1)

    def _get_epsilon(self, game):
        """
        Returns epsilon (random move chance) as decaying e^-x with first
        frac_random*games totally random

        game = current game number
        games = number of training games
        frac_random = fraction of games with epsilon = 1
        final_epsilon = epsilon after games have been played
        min_epsilon = minimum val of epsilon
        """
        if self.mode == "testing":
            self.epsilon = 0
        else:
            self.epsilon = bound(self.min_epsilon, 1, self.final_epsilon**((game - self.frac_random * \
            self.games) / (self.games * (1 - self.frac_random))))
        return self.epsilon

    def _action_space_sample(self):
        """
        Choose a random move
        """
        return random.randint(0, self.num_actions - 1)

    def _matrixify(self, data):
        """
        Change data into matrix format
        """
        return float2mat(data, self.env_size)

class NNConfig():
    """
    mode:
    "training" -> works as normal
    "testing" -> doesn't explore (epsilon = 0)
    """
    def __init__(self,
                mode = "training",
                gamma = 0.6,
                mem_max_size = 1000,
                minibatch_size = 32,
                frac_random = 0.1,
                final_epsilon = 0.01,
                min_epsilon = 0.01
                ):

        self.mode = mode
        self.gamma = gamma
        self.mem_max_size = mem_max_size
        self.minibatch_size = minibatch_size
        self.frac_random = frac_random
        self.final_epsilon = final_epsilon
        self.min_epsilon = min_epsilon

################################# External Functions/Classes ##############################

# Tuple class which contains details of an experience
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'new_state', 'done'])

class ReplayMemory:
    """
    Holds experiences in a deque and returns randomly sampled experiences for replay
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def len(self):
        return len(self.buffer) 

    def append(self, experience):
        self.buffer.append(experience)
  
    def replay_sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, new_states, dones = zip(*[self.buffer[index] for index in indices])
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(dones)

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
        games = 100)
    nn_config = NNConfig(
        mode="training",
        mem_max_size=1000,
        minibatch_size=32,
        frac_random=0.1,
        final_epsilon=0.01,
        min_epsilon=0.01)
    cnn = CNN(gameconfig,nn_config)
    print("Finished")

if __name__ == "__main__":
    main()
