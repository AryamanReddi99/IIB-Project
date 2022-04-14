import os
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.optimizers import Adam
from pkg.plot import *
from pkg.q import *
from pkg.dqn import *
from nim_env_DQNvDQN import *

store_img = True
store_model = False
sample_replay = False  # sample memory or learn from every game
os.chdir(os.path.dirname(__file__))

i = 3
n = 20
env = nim_env_DQNvDQN(i, n)
post_reward = 1
input_dim = env.observation_space_n
n_actions = env.action_space_n

# Hyperparameters
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.995
minibatch_size = 32
replay_memory = []  # replay memory holds s, a, r, s'
mem_max_size = 1000


# Create or load model
if store_model:
    fn = "model_random"
    try:
        model = load_model(fn)
        print("model loaded")
    except:
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(n_actions, activation="softmax"))
        model.compile(optimizer=Adam(), loss="mse")
else:
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_actions, activation="linear"))
    model.compile(optimizer=Adam(), loss="mse")

# Replay function
def replay_sample(replay_memory, minibatch_size=32):
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)
    s_l = np.array(list(map(lambda x: x["s"], minibatch)))
    a_l = np.array(list(map(lambda x: x["a"], minibatch)))
    r_l = np.array(list(map(lambda x: x["r"], minibatch)))
    sprime_l = np.array(list(map(lambda x: x["sprime"], minibatch)))
    done_l = np.array(list(map(lambda x: x["done"], minibatch)))
    qvals_sprime_l = model.predict(sprime_l)
    target_f = model.predict(s_l)  # includes the other actions, states
    # q-update
    for i, (s, a, r, qvals_sprime, done) in enumerate(
        zip(s_l, a_l, r_l, qvals_sprime_l, done_l)
    ):
        if not done:
            target = r + gamma * np.max(qvals_sprime)
        else:
            target = r
        target_f[i][a] = target
    model.fit(s_l, target_f, epochs=1, verbose=0)
    return model


def replay_game(replay_memory):
    s_l = np.array(list(map(lambda x: x["s"], replay_memory)))
    a_l = np.array(list(map(lambda x: x["a"], replay_memory)))
    r_l = np.array(list(map(lambda x: x["r"], replay_memory)))
    sprime_l = np.array(list(map(lambda x: x["sprime"], replay_memory)))
    done_l = np.array(list(map(lambda x: x["done"], replay_memory)))
    qvals_sprime_l = model.predict(sprime_l)
    target_f = model.predict(s_l)  # includes the other actions, states
    # q-update
    for i, (s, a, r, qvals_sprime, done) in enumerate(
        zip(s_l, a_l, r_l, qvals_sprime_l, done_l)
    ):
        if not done:
            target = r + gamma * np.max(qvals_sprime)
        else:
            target = r
        target_f[i][a] = target
    model.fit(s_l, target_f, epochs=1, verbose=0)
    return model


# For plotting metrics
trials = 100
window_size = 10  # n episodes per batch
rewards = np.zeros(trials)  # +1 = p1, -1 = p2
table_abs = []
optimal_table = False

for trial in tqdm(range(trials)):
    [state, turn] = env.reset()
    done = False
    moves_1 = []
    moves_minus_1 = []
    while not done:
        # epsilon update
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        # choose move
        qvals_s = model.predict(state.reshape(1, input_dim))
        if random.uniform(0, 1) < epsilon:
            action = env.action_space_sample()  # Explore
        else:
            action = np.argmax(qvals_s)  # Exploit

        [next_state, turn], reward, done = env.step(action + 1)

        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)

        memory_dict = {
            "s": state,
            "a": action,
            "r": reward,
            "sprime": next_state,
            "done": done,
        }

        if turn == -1:
            moves_1.append(memory_dict)
        elif turn == 1:
            moves_minus_1.append(memory_dict)
        state = next_state

    if turn == 1:  # player 1 won
        for move_dict in moves_1:
            move_dict["r"] = post_reward
        for move_dict in moves_minus_1:
            move_dict["r"] = -post_reward
    if turn == -1:  # player -1 won
        for move_dict in moves_1:
            move_dict["r"] = -post_reward
        for move_dict in moves_minus_1:
            move_dict["r"] = post_reward

    replay_memory.extend(moves_1)
    replay_memory.extend(moves_minus_1)

    if sample_replay:
        if len(replay_memory) > minibatch_size:
            model = replay_sample(replay_memory, minibatch_size)
    else:
        game_memory = moves_1 + moves_minus_1
        model = replay_game(game_memory)

    rewards[trial] = turn

    if trial % 10 == 0:
        q_table = model_to_table(model)
        evaluator = evaluate_q_table(i, n, q_table)
        table_abs.append(evaluator.abs_vals())
        if not optimal_table:
            if evaluator.evaluate_q_table():
                optimal_table = True
                optimal_iter = trial

###################### DIAGNOSTICS #########################
evaluator = evaluate_q_table(i, n, q_table)
if not evaluator.evaluate_q_table():
    print("final faulty q-table: ", evaluator.faulty_rows())
    for faulty_row in evaluator.faulty_rows():
        print(q_table[faulty_row])


###################### PLOTTING #########################
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
plt.title(f"DQN self-play", fontsize=20)
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

print(f"Training finished.\n")
