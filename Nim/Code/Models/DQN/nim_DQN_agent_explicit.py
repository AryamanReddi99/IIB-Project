import os
import pprint
import random
import numpy as np
import pprint
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,save_model
from keras.layers import Dense
from keras.optimizers import Adam
from nim_env_DQN import *
from evaluate_q_table import *

i=3
n=20
opponent = scalable_player(1)
#opponent = random_player()
env = nim_env_DQN(i,n,opponent,first_player="opponent")

# Hyperparameters
alpha = 0.8
gamma = 0.8
epsilon = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99
minibatch_size = 32
replay_memory = [] # replay memory holds s, a, r, s'
mem_max_size = 1000


# Create or load model
fn = "model_random"
try:
    model = load_model(fn)
    print("model loaded")
except:
    n_actions = env.action_space_n
    input_dim = 1
    model = Sequential()
    model.add(Dense(64, input_dim = input_dim , activation = 'relu', bias_initializer='zeros'))
    model.add(Dense(32, activation = 'relu', bias_initializer='zeros'))
    model.add(Dense(n_actions, activation = 'linear'))
    model.compile(optimizer=Adam(), loss = 'mse')

# Replay function
def replay_sample(replay_memory, minibatch_size=32):
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

def replay_game(replay_memory):
    s_l =      np.array(list(map(lambda x: x['s'], replay_memory)))
    a_l =      np.array(list(map(lambda x: x['a'], replay_memory)))
    r_l =      np.array(list(map(lambda x: x['r'], replay_memory)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], replay_memory)))
    done_l   = np.array(list(map(lambda x: x['done'], replay_memory)))
    qvals_sprime_l = model.predict(sprime_l)
    target_f = model.predict(s_l) # includes the other actions, states
    # q-update
    for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
        if not done:  target = r + gamma * np.max(qvals_sprime)
        else:         target = r
        target_f[i][a] = target
    model.fit(s_l,target_f, epochs=1, verbose=0)
    return model


# For plotting metrics
trials  = 1000
batch_size = 10 # n games per batch
scores = [[] for i in range(trials//batch_size)]
for trial in range(trials):
    state = env.reset()
    reward = 0
    done = False
    game_memory=[]
    while not done:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)
        qvals_s = model.predict(np.array([state]))
        if random.uniform(0, 1) < epsilon:
            action = env.action_space_sample() # Explore action space
        else:
            action = np.argmax(qvals_s) # Exploit learned values

        next_state, reward, done = env.step(action+1) 
        
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)

        memory_dict = {
                "s":state,
                "a":action,
                "r":reward,
                "sprime":next_state,
                "done":done}
        
        game_memory.append(memory_dict)
        state = next_state
    #for i in range(len(game_memory)):
    #    game_memory[i]["r"] = reward
    #model=replay_game(game_memory)
    replay_memory.extend(game_memory)
    model=replay_sample(replay_memory, minibatch_size = minibatch_size)
    scores[trial//batch_size].append(reward)
    if not trial%10:  print(trial)
save_model(model, fn)


###################### PLOTTING #########################
y = [np.mean(batch) for batch in scores]
with open("results.csv", "a") as myfile:
    writer = csv.writer(myfile,lineterminator = '\n')
    for item in y:
        writer.writerow([item])
myfile.close()
y_total = []
with open("results.csv", "r") as myfile:
    reader = csv.reader(myfile,lineterminator='\n',delimiter=",")
    for row in reader:
        y_total.extend(row)
myfile.close()
x=np.arange(0,len(y_total))

plt.plot(x,y_total,label=f"Average score over batches of {batch_size} games")
plt.xlabel("Trials")
plt.ylabel(f"Average score")
plt.legend()
plt.show()

print(f"Training finished.\n")