import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
from nim_env import *

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        #self.action_space = action_space
        #self.state_space = state_space

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.8
        self.learning_rate = 0.01

        #self.tau = .125
        #self.model        = self.create_model()
        #self.target_model = self.create_model()
        self.batch_size = 8 # 
        self.model = self.build_model()

    def build_model(self):
        model   = Sequential()
        state_shape  = self.env.state.shape # returns (2,)
        model.add(Dense(32, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.env.action_space_n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon: # chance of exploration instead of argmax policy
            return self.env.action_space_sample()
        return np.argmax(self.model.predict(state)[0]) #predict func returns ordered list

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size) # random sample of memories
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env  = nim_env(3,20)
    trials  = 500
    trial_len = 1000
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    test_batches = 5
    scores = [[] for i in range(trials//test_batches)]
    steps = []
    for trial in range(trials):
        cur_state = env.reset().reshape(1,2)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action+1)
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            cur_state = new_state
            if done:
                break
        dqn_agent.replay()     # internally iterates model
        scores[trial//test_batches].append(reward)
        print("")
    y = [np.mean(batch) for batch in scores]
    x = np.arange(trials//test_batches)
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    main()