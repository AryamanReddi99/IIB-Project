import gym
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make("CartPole-v1")

## Building the nnet that approximates q 
n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
model = Sequential()
model.add(Dense(64, input_dim = input_dim , activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_actions, activation = 'linear'))
model.compile(optimizer=Adam(), loss = 'mse')

def replay(replay_memory, minibatch_size=32):
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    s_l =      np.array(list(map(lambda x: x['s'], minibatch)))
    a_l =      np.array(list(map(lambda x: x['a'], minibatch)))
    r_l =      np.array(list(map(lambda x: x['r'], minibatch)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
    done_l   = np.array(list(map(lambda x: x['done'], minibatch)))
    qvals_sprime_l = model.predict(sprime_l)
    target_f = model.predict(s_l) # includes the other actions, states
    # q-update
    for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
        if not done:  target = r + gamma * np.max(qvals_sprime)
        else:         target = r
        target_f[i][a] = target
    model.fit(s_l,target_f, epochs=1, verbose=0)
    return model


################## RUN SIMULATIONS ###################################


n_episodes = 2500
gamma = 0.99
epsilon = 0.9
minibatch_size = 32
r_sums = []  # stores rewards of each epsiode 
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 100000
for n in range(n_episodes): 
    s = env.reset()
    done=False
    r_sum = 0
    while not done: 
        # Uncomment this to see the agent learning
        # env.render()
        # Feedforward pass for current state to get predicted q-values for all actions 
        qvals_s = model.predict(s.reshape(1,4))
        # Choose action to be epsilon-greedy
        if np.random.random() < epsilon:  
            a = env.action_space.sample()
        else:                             
            a = np.argmax(qvals_s); 
        # Take step, store results 
        sprime, r, done, _ = env.step(a)
        r_sum += r 
        # add to memory, respecting memory buffer limit 
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({
                "s":s,
                "a":a,
                "r":r,
                "sprime":sprime,
                "done":done})
        # Update state
        s=sprime
        # Train the nnet that approximates q(s,a), using the replay memory
        model=replay(replay_memory, minibatch_size = minibatch_size)
        # Decrease epsilon until we hit a target threshold 
    if epsilon > 0.1:      
        epsilon -= 0.001
    #print("Total reward:", r_sum)
    r_sums.append(r_sum)
    if n % 100 == 0: print(n)