import csv
import numpy as np
import matplotlib.pyplot as plt

b = []
with open("results.csv", "r") as myfile:
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