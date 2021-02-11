import random
import keras.models
import numpy as np

def squarify(M,val=0):
    # pads matrix with 0s to make it square
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)


def model_to_table(model):
    """
    convert model predictions to q-table
    easier to test for optimality
    """
    input_dim = model.layers[0].input_shape[-1] # input shape of model
    output_dim = model.layers[-1].output_shape[-1] # action space
    q_table = np.zeros((input_dim,output_dim))
    for i in range(input_dim):
        state = np.zeros(input_dim)
        state[i] = 1
        prediction = model.predict(state.reshape(1,input_dim))
        q_table[i] = prediction
    return q_table

def replay_sample(replay_memory,model, minibatch_size=32):
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

def replay_game(replay_memory, model):
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


class nim_env_DQNvDQN():
    def __init__(self,i,n,max_n,max_i):
        self.i = i
        self.n = n
        # matrix dimensions
        self.max_n = max_n
        self.max_i = max_i
        self.observation_space_n = self.n + 1 + i
        self.tot = 0 # total
        self.diff = n # n-s
        self.turn = 1
        self.done=False       
        self.action_space = np.array([i for i in range(max_i)]) # creates action space of possible moves: e.g. 0,1,2
        self.action_space_n = len(self.action_space)
    def reset(self):
        self.done=0
        self.tot=0
        self.diff = self.n
        self.turn = 1
        return self.update_state()
    def update_state(self):
        if self.done:
            # game is over
            self.diff_vec = np.ones(self.max_n)
            self.state_vec = np.zeros((self.max_i+1,self.max_n+1))
            self.state_vec[0,0] = 1
            self.state = [self.state_vec,self.turn]
            return self.state

        # tot_vec
        self.tot_vec = np.zeros(self.max_n) # total of game, vectorised e.g. [1 0 0 0 0]
        self.tot_vec[self.tot] = 1
        # i_vec
        self.i_vec = np.zeros(self.max_n)
        self.i_vec[self.i] = 1
        # state_vec
        self.state_vec = np.zeros((self.max_i+1,self.max_n+1))
        self.state_vec[self.i,self.diff] = 1

        self.state = [self.state_vec,self.turn]
        #self.state = [self.tot_vec,self.turn]
        return self.state
    def action_space_sample(self):
        return random.choice(self.action_space)
    def step(self,action):
        self.turn *= -1
        if action > self.i: # invalid move, lose immediately
            reward = 0 # handle rewards outside
            self.done=True
            return self.update_state(), reward, self.done
        
        self.tot += action
        self.diff = self.n - self.tot
        
        if self.tot<=self.n:
            reward = 0 # handle rewards outside
            self.done=False
        else:
            reward = 0
            self.done=True
        return self.update_state(), reward, self.done