import numpy as np
from keras.models import load_model,save_model

model=load_model("model_random_prop_reward")
weights, biases = model.layers[0].get_weights()
print(model.predict(np.array([1])))
