import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nim_env_Q import *
from scipy.interpolate import make_interp_spline
from pkg.plot import *
from pkg.q import *

np.random.seed(0)

store_img = False
os.chdir(os.path.dirname(__file__))

i=3
n=20
skill = 1
opponent = scalable_player(skill)
env = nim_env_Q(i,n,opponent,first_player="random")

q_table = np.zeros([(env.n)+4, env.action_space_n])

# Hyperparameters
alpha = 0.8
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99

# For plotting metrics
trials  = 1000
window_size = 10 # n games per batch
rewards = np.zeros(trials) # +1 = we won, -1 = we lost
optimal_table = False
table_abs = []

for trial in tqdm(range(trials)):
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

    evaluator = evaluate_q_table(i,n,q_table)
    table_abs.append(evaluator.abs_vals())
    if not optimal_table and evaluator.evaluate_q_table():
            optimal_table = True
            optimal_iter = trial

    rewards[trial] = reward

# Q-table diagnostics
evaluator = evaluate_q_table(i,n,q_table)
if not evaluator.evaluate_q_table():
    print("final faulty q-table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table[faulty_row])

########## PLOTTING #########
wins = [1 if reward>0 else 0 for reward in rewards] # list of wins, 1=win,0=loss
wins_dis = movav_dis(wins) # discrete win avg over batches of 10
rewards_dis = movav_dis(rewards) # non-overlapping moving average over rewards - still quite jerky
rewards_dis_con = movav_con(rewards_dis) # continuous movav over rewards_dis
xspl,yspl = spline(rewards_dis_con,end=trials) # spline movav for smoothness

# plot win rate results
fig1 = plt.figure(figsize=(8.0, 5.0))
if optimal_table:
    plt.axvline(x=optimal_iter, color='r', linestyle='--', linewidth = 3,label=f"optimal policy learned at iteration {optimal_iter}")
plt.plot(xspl,yspl, linewidth = 3.5)
plt.xlabel("Trials", fontsize=20)
plt.ylabel(f"Average reward", fontsize=20)
plt.title("Q-Learner vs Random", fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()

# plot q-table value results
fig2 = plt.figure(figsize=(8.0, 5.0))
if optimal_table:
    plt.axvline(x=optimal_iter, color='r', linestyle='--', linewidth = 3,label=f"optimal policy learned at iteration {optimal_iter}")
plt.plot(table_abs, linewidth = 3.5)
plt.xlabel("Trials", fontsize=20)
plt.ylabel(f"$\Sigma_(s, a) |Q(s, a)|$", fontsize=20)
plt.title("Q-Learner vs Random", fontsize=20)
plt.legend(loc='lower right', fontsize=20)
plt.show()


if store_img:
    fig1.savefig(f"Images/win_rate_{skill}.jpg", dpi = 100)
    fig2.savefig(f"Images/table_abs_{skill}.jpg", dpi = 100)


print(f"Training finished.\n")