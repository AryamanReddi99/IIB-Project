import keras.models
import numpy as np


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