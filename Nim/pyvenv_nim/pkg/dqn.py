import collections
import datetime
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import TensorBoard
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
    - create target model
    """
    def __init__(self, gameconfig, nn_config):
        ## gameconfig
        self.i = gameconfig.i
        self.n = gameconfig.n
        self.games = gameconfig.games
        self.start_player = gameconfig.start_player
        self.max_i = gameconfig.max_i
        self.max_n = gameconfig.max_n

        ## nn_config
        self.mode = nn_config.mode
        self.gamma = nn_config.gamma
        self.mem_max_size = nn_config.mem_max_size
        self.minibatch_size = nn_config.minibatch_size
        self.epoch_size = nn_config.epoch_size
        self.num_filters = nn_config.num_filters
        self.kernel_regulariser = nn_config.kernel_regulariser
        self.kernel_activation = nn_config.kernel_activation
        self.truncate_i = nn_config.truncate_i
        self.frac_random = nn_config.frac_random
        self.final_epsilon = nn_config.final_epsilon
        self.min_epsilon = nn_config.min_epsilon
        self.learning_rate = nn_config.learning_rate
        self.tensorboard = nn_config.tensorboard
        self.epochs = nn_config.epochs
        self.target_model_iter = nn_config.target_model_iter
        self.reward_mode = nn_config.reward_mode
        self.optimal_override = nn_config.optimal_override
        self.test_divisors = nn_config.test_divisors

        # Modulo i+1
        self.max_i_mod = self.max_i + 1

        ## Buffers
        # stores last 2 sets of model-readable states
        self.state_buffer = collections.deque(maxlen=3)
        # stores experiences from one game
        self.game_buffer = []
        # stores experiences
        self.replay_buffer = collections.deque(maxlen=self.mem_max_size)

        # Tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    def update_state_buffer(self, i, t, turn):
        game_state = {
                    "i":i,
                    "t":t,
                    "turn":turn} 
        self.state_buffer.append(game_state)

    def update_game_buffer_2(self, action, reward, done):
        """
        Update game buffer using previous 2 states
        """
        experience = Experience(self.state_buffer[-2], action, reward, self.state_buffer[-1], done)
        self.game_buffer.append(experience) 

    def update_game_buffer_3(self, action, reward, done):
        """
        Update game buffer using previous 3 states
        """
        if len(self.state_buffer)==3:
            experience = Experience(self.state_buffer[-3], action, reward, self.state_buffer[-1], done)
            self.game_buffer.append(experience) 
    
    def act(self, game):
        """
        Looks at most recent state in state buffer and chooses move
        """
        state = self.state_buffer[-1]
        i = state["i"]
        t = state["t"]
        # override
        if self.optimal_override:
            action = optimal_play(i,t)
        # explore
        elif random.uniform(0,1) < self._get_epsilon(game):
            action = self._action_space_sample(i,t)
        # exploit
        else:
            # make machine-readable state
            action = self._predict(i, t)
        return action

    def update_experiences(self, reward_list):
        self._update_game_buffer(reward_list)
        self.replay_buffer.extend(self.game_buffer)
        self.game_buffer = []
        self.state_buffer.clear()

    def train(self, move_total):
        if self.mode == "testing":
            return
        if len(self.replay_buffer) < self.epoch_size:
            return
        if move_total%self.target_model_iter==0:
            self._update_target_model()
        self.replay_sample = self._get_replay_sample()
        self._experience_replay()

    def total_optimality(self):
        """
        Returns mean optmality of network for all test divisors
        """
        divisor_optimalities = list(map(lambda x : self._i_table_optimality(x), self.test_divisors))
        return np.mean(divisor_optimalities)

    def _predict(self, i, t):
        """
        Model's chosen action for a given i,t
        """
        readable_state = self._readable_state_single(self._readable_state(i,t))
        qvals = self.model.predict(readable_state)
        action = np.argmax(qvals) + 1
        return action

    def _experience_replay(self):
        """
        Trains a model on q(s,a) of sampled experiences
        """

        # Recieve SARSD training data as 5 numpy arrays
        states, actions, rewards, new_states, dones = self.replay_sample

        # Reshape states and new states into batches
        states_readable_batch = np.array(list(map(lambda state: self._readable_state(state["i"], state["t"]), states)))
        new_states_readable_batch = np.array(list(map(lambda state: self._readable_state(state["i"], state["t"]), states)))

        # Predicted state q values
        target_qvals = self.model.predict(states_readable_batch)

        # Predicted new state q values
        new_states_qvals = self.target_model.predict(new_states_readable_batch)

        # Train on each experience
        for i, (state_readable,action,reward,new_state_qvals,done) in enumerate(zip(states_readable_batch,actions,rewards,new_states_qvals,dones)): 
            if done:
                target_qval = reward
            else:
                target_qval = reward + self.gamma * np.max(new_state_qvals)
            target_qvals[i][action-1] = target_qval
        self._cnn_fit(states_readable_batch, target_qvals, self.epochs)

    def _cnn_fit(self,x,y,epochs):
        if self.tensorboard:
            history = self.model.fit(
                    x=x, 
                    y=y, 
                    batch_size=self.minibatch_size, 
                    epochs=epochs, 
                    verbose=1,
                    callbacks=[self.tensorboard_callback])
        else:
            history = self.model.fit(
                    x=x, 
                    y=y, 
                    batch_size=self.minibatch_size, 
                    epochs=epochs, 
                    verbose=1)
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        self.loss.append(loss)
        self.accuracy.append(accuracy)

    def _get_replay_sample(self):
        indices = np.random.choice(len(self.replay_buffer), self.epoch_size, replace=False)
        states, actions, rewards, new_states, dones = zip(*[self.replay_buffer[index] for index in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(new_states), np.array(dones)

    def _create_sequential(self):
        """
        creates sequential cnn
        """
        model = Sequential(name="CNN_1D")
        
        # Input, Conv1D
        model.add(Conv1D(filters=self.num_filters, kernel_size=(self.max_i_mod), 
                        input_shape=((self.max_n+1)+2*(self.max_i_mod-1), 2), padding="valid", 
                        strides=(1), activation=self.kernel_activation,
                        kernel_regularizer=regularizers.L1(0.01), name="conv1d"))

        model.add(GlobalMaxPooling1D(name="maxpool"))

        model.add(Dense(self.max_i, name="output", activation='linear'))

        model.compile(
                    loss='mean_squared_error',
                    optimizer=Adam(learning_rate=self.learning_rate),
                    metrics=['accuracy'])
  
        return model

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
    
    def _action_space_sample(self,i,t):
        """
        Choose a random legal move
        """
        return random_play(i,t)

    def _readable_state(self,i,t):
        """
        Create model-friendly input state
        """
        i_mod = i + 1
        t_vec = one_hot(t, self.max_n)
        if self.truncate_i:
            i_mod_vec = one_hot_repeat_truncate(i_mod, self.max_n)
        else:
            i_mod_vec = one_hot_repeat(i_mod, self.max_n)
        stacked = np.vstack((np.flip(t_vec),np.flip(i_mod_vec))).T
        stacked_padded = np.pad(stacked, pad_width=((self.max_i_mod-1,self.max_i_mod-1),(0,0)))
        return stacked_padded

    def _readable_state_single(self, state):
        """
        Adds an extra dimenion to a single readable state
        """
        state_shape = state.shape
        single_state = state.reshape(1, *state_shape)
        return single_state

    def _update_game_buffer(self, reward_list):
        # Final
        if self.reward_mode==0:
            return self.game_buffer
        # Sparse
        if self.reward_mode==1:
            # single learner
            self.game_buffer[-1]._replace(reward=reward_list[self.game_buffer[-1].state["turn"]])
            # double learner
            if len(self.game_buffer)>1 and (self.game_buffer[-1].state["turn"] != self.game_buffer[-2].state["turn"]):
                self.game_buffer[-2]._replace(reward=reward_list[self.game_buffer[-2 ].state["turn"]])
            return self.game_buffer
        # Full
        if self.reward_mode==2:
            self.game_buffer = list(map(lambda x: x._replace(reward=reward_list[x.state["turn"]]),self.game_buffer))
            return self.game_buffer

    def _i_table_optimality(self,i):
        """"
        Gets the Q-value predictions of the network for a certain value of i for dividends upto i_mod
        """
        table = np.zeros([self.i,self.max_i])
        for t, row in enumerate(table):
            readable_state = self._readable_state_single(self._readable_state(i,t))
            qvals = self.model.predict(readable_state)
            row=qvals
        return table_optimality(i, table)


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
                epoch_size = 64,
                num_filters = 10,
                kernel_regulariser = 0.01,
                kernel_activation = 'relu',
                truncate_i = False,
                frac_random = 0.1,
                final_epsilon = 0.01,
                min_epsilon = 0.01,
                learning_rate = 0.001,
                tensorboard = False,
                epochs = 1,
                target_model_iter = 10,
                reward_mode = 0,
                optimal_override = 0,
                test_divisors = 3
                ):
        self.mode = mode
        self.gamma = gamma
        self.mem_max_size = mem_max_size
        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.num_filters = num_filters
        self.kernel_regulariser = kernel_regulariser
        self.kernel_activation = kernel_activation 
        self.truncate_i = truncate_i
        self.frac_random = frac_random
        self.final_epsilon = final_epsilon
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.tensorboard = tensorboard
        self.epochs = epochs
        self.target_model_iter = target_model_iter
        self.test_divisors = test_divisors

        # Reward_mode:
        # 0: Final - only the rewards given by env
        # 1: Sparse - terminal move of either agent
        # 2: Full - terminal rewards propagate through whole game
        self.reward_mode = reward_mode

        # Optimal Override
        # Switch on to force optimal play - use with caution,
        # Only for testing
        self.optimal_override = optimal_override