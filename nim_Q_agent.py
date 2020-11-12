import os
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from nim_env_Q import *
from evaluate_q_table import *
from moving_average import *

class nim_Q_agent():
    def __init__(self,i,n,alpha=0.6,gamma=0.8,epsilon=1,epsilon_min=0.001,epsilon_decay=0.995):
        self.i=i
        self.n=n

        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay

        self.action_space = np.array([i for i in range(i)])
        self.q_table=np.full([(n)+(i+1), len(self.action_space)],-1)
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state["tot"]])
        return action
    def update(self,state,action,reward,next_state):
        old_value = self.q_table[state["tot"], action]
        next_max = np.max(self.q_table[next_state["tot"]])
        new_value = (1 - self.alpha)*old_value + self.alpha*(reward + self.gamma*next_max)
        self.q_table[state["tot"], action] = new_value

i=3
n=20
trials  = 10000
batch_size = 100 # n games per batch
scores = [[] for i in range(trials//batch_size)]
optimal_table_reached = False
optimal_iter = 0

p1 = nim_Q_agent(i,n)
p2 = p1
env = nim_env_Q(3,20,p1,p2,"random",0,-10)

for trial in range(trials):
    state = env.reset()
    done = False
    while not done:
        action = env.current_player.act(state)
        next_state, reward, done = env.step(action+1)
        env.current_player.update(state,action,reward,next_state)
        state = next_state

        # if not optimal_table_reached:
        #     evaluator = evaluate_q_table(i,n,agent.q_table)
        #     if evaluator.evaluate_q_table():
        #         optimal_table_reached = True
        #         optimal_iter = trial

    scores[trial//batch_size].append(reward)

# evaluator = evaluate_q_table(i,n,agent.q_table)
# if not evaluator.evaluate_q_table():
#     print("final faulty q_table: ", evaluator.faulty_rows())
#     for faulty_row in evaluator.faulty_rows():
#         print(q_table[faulty_row])



window_MA = 10
x_MA,y_MA = moving_average(scores,batch_size,trials,window_MA)

y = [np.mean(batch) for batch in scores]
x=np.linspace(0,trials,len(y))
plt.plot(x,y,label=f"Average score over batches of {batch_size} games")
plt.plot(x_MA,y_MA,label=f"Moving average, window = {window_MA}", color="orange")
if optimal_table_reached:
    plt.axvline(x=optimal_iter, color='r', linestyle='-', linewidth = 2,label=f"optimal policy learned at iteration {optimal_iter}")
plt.xlabel("Trials")
plt.ylabel(f"Average score")
#plt.title(f"Q-learner vs random player\ni={i}, n={n}, alpha={alpha}, gamma={gamma}")
plt.legend()
plt.show()

print(f"Training finished.\n")