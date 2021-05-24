import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cum_avg(
    arr,
    x_title="Game",
    y_title="Cumulative Win Rate",
    title="Player vs Player",
    player_names=["Q-Learner", "Scalable"]):
    """
    Displays the cumulative average of some data
    """

    # Data Pre-process
    x = np.arange(1,len(arr)+1)
    p1_wins = np.cumsum(arr)/x
    p0_wins = np.cumsum(1-arr)/x
    
    # Dataframe
    df = pd.DataFrame({x_title:x,player_names[1]:p1_wins,player_names[0]:p0_wins})
    df_melt = pd.melt(df, [x_title], var_name="Player", value_name=y_title)

    # Plotting
    sns.set_theme()
    sns.lineplot(data=df_melt, x=x_title, y=y_title, hue="Player").set(title=title)
    plt.xlim(20,None)

def ma(arr, w=11):
    """
    Returns moving average of array of window size w (must be odd)
    Padds arr and uses valid mode - equivalent to using same, but without dropoff at edges
    """
    # Get dimensions
    arr_len = len(arr)
    padding = int((w-1)/2)

    # Pad array
    arr_padded = np.zeros(arr_len+2*padding)
    arr_padded[:padding]=arr[0]
    arr_padded[padding:-padding]=arr
    arr_padded[-padding:]=arr[-1]
    return np.convolve(arr_padded, np.ones(w), 'valid') / w
