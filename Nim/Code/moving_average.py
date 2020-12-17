import numpy as np

def moving_average(scores,batch_size,trials,window_MA=10):
    y = [np.mean(batch) for batch in scores]
    i = 0
    y_MA = []
    while i < len(y) - window_MA + 1:
        this_window = y[i : i + window_MA]
        window_average = sum(this_window)/window_MA
        y_MA.append(window_average)
        i += 1
    x_MA = np.linspace(0.5*batch_size,trials-0.5*batch_size,len(y_MA))
    return x_MA, y_MA