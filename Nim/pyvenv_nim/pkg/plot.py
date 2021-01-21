import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline,make_interp_spline

def chunkify(x,w=10):
    """
    converts list to lists of w-length lists
    [1,2,3,4,5,6] -> [[1,2,3],[4,5,6]]
    """
    chunks = [x[i:i+w] for i in range(0, len(x), w)] # break into smaller lists
    return chunks

def movav_dis(x,w=10):
    """
    returns non-overlapping moving average of x
    w = window length
    """
    chunks = chunkify(x,w)
    chunks_avg = [np.mean(chunk) for chunk in chunks] # mean of each chunk
    return chunks_avg

def movav_con(x,w=10,mode='valid'):
    """
    returns continuous moving average of np array x with window w 
    mode = 'same' maintains array length
    mode = 'valid' only convolves where there is overlap
    w = window length
    """
    try:
        return np.convolve(x, np.ones(w), mode) / w
    except:
        try:
            return np.convolve(np.array(x), np.ones(w), mode) / w
        except:
            raise TypeError("x needs to be a list or np.array")

def cumav(x):
    """
    returns cumulative average of np array x
    """
    try:
        return np.cumsum(x)/np.arange(1,len(x)+1)
    except:
        try:
            return np.cumsum(np.array(x))/np.arange(1,len(x)+1)
        except:
            raise TypeError("x needs to be a list or np.array")

def spline(y,end,points=300):
    """
    returns spline smoothed y
    end = max on x axis
    """
    xtrue = np.linspace(0,end,len(y))
    xspl = np.linspace(0, end, points) # points represents number of points to make between T.min and T.max
    spl = make_interp_spline(xtrue, y, k=3)  # BSpline y over xspl
    yspl = spl(xspl)
    return xspl, yspl

def plot_from_csv(fn):
    """
    Outputs results of DQN training to a csv
    """
    b = []
    with open("fn", "r") as myfile:
        reader = csv.reader(myfile,lineterminator = '\n',delimiter=",")
        for row in reader:
            b.extend(row)
    myfile.close()
    c = [0.1*(float(item)*0.1+1) for item in b]
    d = np.cumsum(c)
    x_MA = np.linspace(0,len(c),len(c))
    plt.plot(x_MA,d)
    plt.xlabel("Trials")
    plt.ylabel("Performance")
    plt.title("DQN vs Scalable Player")
    plt.show()
    print("finished")