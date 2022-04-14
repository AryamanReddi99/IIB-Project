import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nim_env_QvQ import *
from scipy.interpolate import make_interp_spline
from pkg.plot import *
from pkg.q import *


store_img = True
os.chdir(os.path.dirname(__file__))

i = 3
n = 20
env = nim_env_QvQ(i, n)
post_reward = 1

q_table = np.zeros([(env.n) + 1 + i, env.action_space_n])

# Hyperparameters
alpha = 0.5
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99

# Plotting metrics
trials = 200
window_size = 10  # n episodes per batch
rewards = np.zeros(trials)  # +1 = p1, -1 = p2
table_abs = []
row_abs = []
optimal_table = False

for trial in tqdm(range(trials)):
    state = env.reset()
    done = False
    moves_1 = []
    moves_minus_1 = []
    while not done:
        # episilon update
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        # choose move
        if random.uniform(0, 1) < epsilon:
            action = env.action_space_sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        [next_state, turn], reward, done = env.step(action + 1)  # dummy reward

        if turn == -1:
            moves_1.append([state, action, next_state])
        elif turn == 1:
            moves_minus_1.append([state, action, next_state])
        state = next_state
    if turn == 1:  # player 1 won
        for state, action, new_state in moves_1:
            q_table[state, action] = q_table[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table[new_state])
                - q_table[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table[state, action] = q_table[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table[new_state])
                - q_table[state, action]
            )
    if turn == -1:  # player -1 won
        for state, action, new_state in moves_1:
            q_table[state, action] = q_table[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table[new_state])
                - q_table[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table[state, action] = q_table[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table[new_state])
                - q_table[state, action]
            )

    rewards[trial] = turn

    evaluator = evaluate_q_table(i, n, q_table)
    table_abs.append(evaluator.abs_vals())
    if not optimal_table:
        if evaluator.evaluate_q_table():
            optimal_table = True
            optimal_iter = trial

    if trial % 100 == 0:
        row_abs.append(evaluator.rows_abs_vals())

# Diagnostics
evaluator = evaluate_q_table(i, n, q_table)
if not evaluator.evaluate_q_table():
    print("final faulty q-table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table[faulty_row])


# Plotting
rewards_lists = chunkify(rewards, 10)
win_prob_1 = [np.count_nonzero(window == 1) for window in rewards]
win_prob_minus_1 = [np.count_nonzero(window == -1) for window in rewards]


fig1 = plt.figure(figsize=(8.0, 5.0))
xspl_1, yspl_1 = spline(win_prob_1, end=trials)
xspl_minus_1, yspl_minus_1 = spline(win_prob_minus_1, end=trials)
if optimal_table:
    plt.axvline(
        x=optimal_iter,
        color="r",
        linestyle="--",
        linewidth=3,
        label=f"optimal policy learned at iteration {optimal_iter}",
    )
plt.plot(xspl_1, yspl_1, linewidth=2, label="Player 1 win prob")
plt.plot(xspl_minus_1, yspl_minus_1, linewidth=2, label="Player 2 win prob")
plt.xlabel("Trials", fontsize=20)
plt.ylabel(f"Win %", fontsize=20)
plt.title(f"Q-Learner self-play", fontsize=20)
plt.legend(loc="lower right", fontsize=15)
plt.show()

fig2 = plt.figure(figsize=(8.0, 5.0))
if optimal_table:
    plt.axvline(
        x=optimal_iter,
        color="r",
        linestyle="--",
        linewidth=3,
        label=f"optimal policy learned at iteration {optimal_iter}",
    )
plt.plot(table_abs, linewidth=3.5)
plt.xlabel("Trials", fontsize=20)
plt.ylabel(f"$\Sigma_(s, a) |Q(s, a)|$", fontsize=20)
plt.title(f"Q-Learner self-play")
plt.legend(loc="lower right", fontsize=20)
plt.show()

if store_img:
    fig1.savefig(f"Images/win_metrics_{n}.jpg", dpi=600)
    fig2.savefig(f"Images/table_metrics_{n}.jpg", dpi=600)
print("End")
