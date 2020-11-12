import os
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from nim_env_Q import *

def clear():
    os.system( 'cls' )

env = nim_env_Q(3,21,first_player="agent")

q_table = np.zeros([(env.n)+4, env.action_space_n])

# Hyperparameters
alpha = 0.5
gamma = 0.4
epsilon = 1.0
epsilon_min = 0.00001
epsilon_decay = 0.995

# For plotting metrics
trials  = 2000
test_batches = 100 # n games per batch
x=np.arange(test_batches,trials+test_batches,test_batches)
scores = [[] for i in range(trials//test_batches)]

q_3_memory= np.zeros((3,trials))

for trial in range(trials):
    state = env.reset()
    epochs, reward, = 0, 0
    done = False
    
    while not done:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space_sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done = env.step(action+1) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

    #q_3_memory[0][trial] = q_table[3][0]
    #q_3_memory[1][trial] = q_table[3][1]
    #q_3_memory[2][trial] = q_table[3][2]
    # print(trial)
    # clear()
    scores[trial//test_batches].append(reward)

y = [np.mean(batch) for batch in scores]
plt.plot(x,y,label=f"Average score over batches of {test_batches} games")

window_size = 3 #compute MA
i = 0
moving_averages = []
while i < len(y) - window_size + 1:
    this_window = y[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1
x_MA = np.linspace(test_batches,trials,len(moving_averages))
plt.plot(x_MA,moving_averages,label=f"Moving average, window = {window_size}")
plt.xlabel("Trials")
plt.ylabel(f"Average score")
plt.legend()
plt.show()

print(f"Training finished.\n")