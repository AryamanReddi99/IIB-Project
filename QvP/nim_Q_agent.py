import os
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from nim_env_Q import *
from evaluate_q_table import *

i=3
n=20
opponent = scalable_player(1)
#opponent = random_player()
env = nim_env_Q(i,n,opponent,first_player="opponent")

q_table = np.zeros([(env.n)+4, env.action_space_n])

# Hyperparameters
alpha = 0.1
gamma = 0.1
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.995

# For plotting metrics
trials  = 1000
batch_size = 10 # n games per batch
x=np.arange(batch_size,trials+batch_size,batch_size)
scores = [[] for i in range(trials//batch_size)]
optimal_table_reached = False

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

        if not optimal_table_reached:
            evaluator = evaluate_q_table(i,n,q_table)
            if evaluator.evaluate_q_table():
                optimal_table_reached = True
                optimal_iter = trial

    scores[trial//batch_size].append(reward)

evaluator = evaluate_q_table(i,n,q_table)
if not evaluator.evaluate_q_table():
    print("final faulty q_table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table[faulty_row])


y = [np.mean(batch) for batch in scores]
plt.plot(x,y,label=f"Average score over batches of {batch_size} games")

window_size = 3 #compute MA
i = 0
moving_averages = []
while i < len(y) - window_size + 1:
    this_window = y[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1
x_MA = np.linspace(batch_size,trials,len(moving_averages))
plt.plot(x_MA,moving_averages,label=f"Moving average, window = {window_size}")
if optimal_table_reached:
    plt.axvline(x=optimal_iter, color='r', linestyle='-', linewidth = 2,label=f"optimal policy learned at iteration {optimal_iter}")
plt.xlabel("Trials")
plt.ylabel(f"Average score")
plt.legend()
plt.show()

print(f"Training finished.\n")