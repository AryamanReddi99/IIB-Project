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

################################ FULL ##########################

i = 3
n = 20
env = nim_env_QvQ(i, n)
post_reward = 1

q_table_full = np.zeros([(env.n) + 1 + i, env.action_space_n])

# Hyperparameters
alpha = 0.6
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99

# Plotting metrics
trials = 200
window_size = 10  # n episodes per batch
rewards = np.zeros(trials)  # +1 = p1, -1 = p2
table_abs_full = []
row_abs = []
optimal_table_full = False

for trial in tqdm(range(trials)):
    [state, turn] = env.reset()
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
            action = np.argmax(q_table_full[state])  # Exploit

        [next_state, turn], reward, done = env.step(action + 1)  # dummy reward

        if turn == -1:
            moves_1.append([state, action, next_state])
        elif turn == 1:
            moves_minus_1.append([state, action, next_state])
        state = next_state
    if turn == 1:  # player 1 won
        for state, action, new_state in moves_1:
            q_table_full[state, action] = q_table_full[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table_full[new_state])
                - q_table_full[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table_full[state, action] = q_table_full[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table_full[new_state])
                - q_table_full[state, action]
            )
    if turn == -1:  # player -1 won
        for state, action, new_state in moves_1:
            q_table_full[state, action] = q_table_full[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table_full[new_state])
                - q_table_full[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table_full[state, action] = q_table_full[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table_full[new_state])
                - q_table_full[state, action]
            )

    rewards[trial] = turn

    evaluator = evaluate_q_table(i, n, q_table_full)
    table_abs_full.append(evaluator.abs_vals())
    if not optimal_table_full:
        if evaluator.evaluate_q_table():
            optimal_table_full = True
            optimal_iter_full = trial

    if trial % 100 == 0:
        row_abs.append(evaluator.rows_abs_vals())

# Diagnostics
evaluator = evaluate_q_table(i, n, q_table_full)
if not evaluator.evaluate_q_table():
    print("final faulty q-table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table_full[faulty_row])

################################ SPARSE ##########################

############################# NOT WORKING -- COME BACK AND FIX THIS ##############################

# i=3
# n=20
# env = nim_env_QvQ(i,n)
# post_reward = 1

# q_table_sparse = np.zeros([(env.n)+1+i, env.action_space_n])

# # Hyperparameters
# alpha = 0.9
# gamma = 0.6
# epsilon = 1.0
# epsilon_min = 0.001
# epsilon_decay = 0.99

# # Plotting metrics
# window_size = 10 # n episodes per batch
# rewards = np.zeros(trials) # +1 = p1, -1 = p2
# table_abs_sparse = []
# row_abs = []
# optimal_table_sparse = False

# for trial in tqdm(range(trials)):
#     [state, turn] = env.reset()
#     done = False
#     moves_1 = []
#     moves_minus_1 = []
#     while not done:
#         # episilon update
#         epsilon *= epsilon_decay
#         epsilon = max(epsilon_min, epsilon)
#         # choose move
#         if random.uniform(0, 1) < epsilon:
#             action = env.action_space_sample() # Explore
#         else:
#             action = np.argmax(q_table_sparse[state]) # Exploit

#         [next_state, turn], reward, done = env.step(action+1) # dummy reward

#         if turn==-1:
#             moves_1.append([state, action, next_state])
#         elif turn==1:
#             moves_minus_1.append([state, action, next_state])
#         state = next_state

#     if turn==1: # player 1 won
#         for i,l in enumerate(moves_1):
#             state,action,new_state = l
#             if i == len(moves_1) - 1:
#                 sparse_reward = 1
#             else:
#                 sparse_reward = 0
#             q_table_sparse[state, action] = q_table_sparse[state, action] + alpha*(sparse_reward + gamma*np.max(q_table_sparse[new_state]) - q_table_sparse[state, action])
#             q_table_full[state, action] = q_table_full[state, action] + alpha*(post_reward + gamma*np.max(q_table_full[new_state]) - q_table_full[state, action])
#         for i,l in enumerate(moves_minus_1):
#             state,action,new_state = l
#             if i == len(moves_minus_1) - 1:
#                 sparse_reward = 1
#             else:
#                 sparse_reward = 0
#             q_table_sparse[state, action] = q_table_sparse[state, action] + alpha*(-1*sparse_reward + gamma*np.max(q_table_sparse[new_state]) - q_table_sparse[state, action])
#     if turn==-1: # player 2 won
#         for i,l in enumerate(moves_1):
#             state,action,new_state = l
#             if i == len(moves_1) - 1:
#                 sparse_reward = 1
#             else:
#                 sparse_reward = 0
#             q_table_sparse[state, action] = q_table_sparse[state, action] + alpha*(-1*sparse_reward + gamma*np.max(q_table_sparse[new_state]) - q_table_sparse[state, action])
#         for i,l in enumerate(moves_minus_1):
#             state,action,new_state = l
#             if i == len(moves_minus_1) - 1:
#                 sparse_reward = 1
#             else:
#                 sparse_reward = 0
#             q_table_sparse[state, action] = q_table_sparse[state, action] + alpha*(sparse_reward + gamma*np.max(q_table_sparse[new_state]) - q_table_sparse[state, action])

#     rewards[trial] = turn

#     evaluator = evaluate_q_table(i,n,q_table_sparse)
#     table_abs_sparse.append(evaluator.abs_vals())
#     if not optimal_table_sparse:
#         if evaluator.evaluate_q_table():
#             optimal_table_sparse = True
#             optimal_iter_sparse = trial

#     if trial%100==0:
#         row_abs.append(evaluator.rows_abs_vals())

# # Diagnostics
# evaluator = evaluate_q_table(i,n,q_table_sparse)
# if not evaluator.evaluate_q_table():
#     print("final faulty q-table: ", evaluator.faulty_rows())
#     for faulty_row in evaluator.faulty_rows():
#         print(q_table_sparse[faulty_row])


########################## PLOTTING ########################################

q_table_sparse = np.zeros([(env.n) + 1 + i, env.action_space_n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99

# Plotting metrics
rewards = np.zeros(trials)  # +1 = p1, -1 = p2
table_abs_sparse = []
row_abs = []
optimal_table_sparse = False

for trial in tqdm(range(trials)):
    [state, turn] = env.reset()
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
            action = np.argmax(q_table_sparse[state])  # Exploit

        [next_state, turn], reward, done = env.step(action + 1)  # dummy reward

        if turn == -1:
            moves_1.append([state, action, next_state])
        elif turn == 1:
            moves_minus_1.append([state, action, next_state])
        state = next_state
    if turn == 1:  # player 1 won
        for state, action, new_state in moves_1:
            q_table_sparse[state, action] = q_table_sparse[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table_sparse[new_state])
                - q_table_sparse[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table_sparse[state, action] = q_table_sparse[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table_sparse[new_state])
                - q_table_sparse[state, action]
            )
    if turn == -1:  # player -1 won
        for state, action, new_state in moves_1:
            q_table_sparse[state, action] = q_table_sparse[state, action] + alpha * (
                -1 * post_reward
                + gamma * np.max(q_table_sparse[new_state])
                - q_table_sparse[state, action]
            )
        for state, action, new_state in moves_minus_1:
            q_table_sparse[state, action] = q_table_sparse[state, action] + alpha * (
                post_reward
                + gamma * np.max(q_table_sparse[new_state])
                - q_table_sparse[state, action]
            )

    rewards[trial] = turn

    evaluator = evaluate_q_table(i, n, q_table_sparse)
    table_abs_sparse.append(evaluator.abs_vals())
    if not optimal_table_sparse:
        if evaluator.evaluate_q_table():
            optimal_table_sparse = True
            optimal_iter_sparse = trial

    if trial % 100 == 0:
        row_abs.append(evaluator.rows_abs_vals())

# Diagnostics
evaluator = evaluate_q_table(i, n, q_table_sparse)
if not evaluator.evaluate_q_table():
    print("final faulty q-table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table_sparse[faulty_row])

# Plotting
fig1 = plt.figure(figsize=(8.0, 5.0))
if optimal_table_full:
    plt.axvline(
        x=50,
        color="r",
        linestyle="--",
        linewidth=3,
        label=f"optimal policy learned at iteration {50}",
    )
if optimal_table_sparse:
    plt.axvline(
        x=102,
        color="blue",
        linestyle="--",
        linewidth=3,
        label=f"optimal policy learned at iteration {102}",
    )
plt.plot(table_abs_full, color="r", linewidth=3.5, label="full rewards")
plt.plot(table_abs_sparse, color="blue", linewidth=3.5, label="sparse rewards")
plt.xlabel("Trials", fontsize=20)
plt.ylabel(f"$\Sigma_(s, a) |Q(s, a)|$", fontsize=20)
plt.title(f"Q-Learner self-play")
plt.legend(loc="lower right", fontsize=20)
plt.show()


if store_img:
    fig1.savefig(f"Images/win_metrics_{n}.jpg", dpi=600)
    # fig2.savefig(f"Images/table_metrics_{n}.jpg", dpi = 600)
print("End")
