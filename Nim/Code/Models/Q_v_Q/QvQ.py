import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from nim_env_QvQ import *
from evaluate_q_table import *

os.chdir(os.path.dirname(__file__))
i=3
n=20
env = nim_env_QvQ(i,n)
post_reward = 1

q_table = np.zeros([(env.n)+4, env.action_space_n])

# Hyperparameters
alpha = 0.1
gamma = 0.1
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99

# Plotting metrics
trials  = 1000
window_size = 10 # n games per batch
winners = [[] for i in range(trials//window_size)]
table_abs_sum = []
optimal_table_reached = False

for trial in range(trials):
    state = env.reset()
    done = False
    turn = 1
    moves_1 = []
    moves_minus_1 = []
    while not done:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space_sample() # Explore
        else:
            action = np.argmax(q_table[state]) # Exploit

        next_state, reward, done = env.step(action+1) # dummy reward

        if turn==1:
            moves_1.append([state, action, next_state])
        else:
            moves_minus_1.append([state, action, next_state])
        turn*=-1 # switch player
        state = next_state
    if turn==1: # player 1 won
        for state, action, new_state in moves_1:
            q_table[state, action] = q_table[state, action] + alpha*(post_reward + gamma*np.max(q_table[new_state]) - q_table[state, action])
        for state, action, new_state in moves_minus_1:
            q_table[state, action] = q_table[state, action] + alpha*(-1*post_reward + gamma*np.max(q_table[new_state]) - q_table[state, action])
    if turn==-1: # player -1 won
        for state, action, new_state in moves_1:
            q_table[state, action] = q_table[state, action] + alpha*(-1*post_reward + gamma*np.max(q_table[new_state]) - q_table[state, action])
        for state, action, new_state in moves_minus_1:
            q_table[state, action] = q_table[state, action] + alpha*(post_reward + gamma*np.max(q_table[new_state]) - q_table[state, action])
    winners[trial//window_size].append(turn)

    evaluator = evaluate_q_table(i,n,q_table)
    table_abs_sum.append(evaluator.abs_vals())
    if not optimal_table_reached:
        if evaluator.evaluate_q_table():
            optimal_table_reached = True
            optimal_iter = trial

x = np.arange(window_size,trials + window_size, window_size)
win_prob_1 = [window.count(1)/len(window) for window in winners]
win_prob_minus_1 = [window.count(-1)/len(window) for window in winners]

fig = plt.figure()
xnew = np.linspace(x.min(), x.max(), 300)  
spl_1 = make_interp_spline(x, win_prob_1, k=3)
power_smooth_1 = spl_1(xnew)
plt.plot(xnew, power_smooth_1, label="Player 1 win prob")
spl_minus_1 = make_interp_spline(x, win_prob_minus_1, k=3)
power_smooth_minus_1 = spl_minus_1(xnew)
plt.plot(xnew, power_smooth_minus_1, label="Player 2 win prob")
if optimal_table_reached:
    plt.axvline(x=optimal_iter, color='r', linestyle='-', linewidth = 2,label=f"optimal policy learned at iteration {optimal_iter}")
plt.xlabel("Trials")
plt.ylabel(f"Win %")
plt.title(f"Q-Learner self-play, n = {n}")
plt.legend()
fig.savefig(f"win_metrics_{n}.jpg")

fig2 = plt.figure()
if optimal_table_reached:
    plt.axvline(x=optimal_iter, color='r', linestyle='-', linewidth = 2,label=f"optimal policy learned at iteration {optimal_iter}")
xnew = np.linspace(0, trials, 300)  
spl_table = make_interp_spline(np.arange(0,trials), table_abs_sum, k=3)
power_smooth_table = spl_table(xnew)
plt.plot(xnew, power_smooth_table)
plt.xlabel("Trials")
plt.ylabel("$\Sigma_(s, a) |Q(s, a)|$")
plt.title(f"Q-Learner self-play, n = {n}")
plt.legend()
fig2.savefig(f"table_metrics_{n}.jpg")

print("End")