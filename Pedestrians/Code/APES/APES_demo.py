from APES.APES import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pylab as pl
from IPython import display
import skvideo.io

perf = {'sub':(2,1),'dom':(2,10),'food':(3,4),'obs':(3,5),'subdir':'W','domdir':'E','mesg':'example'}
game = CreateEnvironment(perf)

agents = [game.agents[i] for i in game.agents]

game.Step()
env_initial = game.BuildImage()

#Execute every time step
agents[0].NextAction = Settings.PossibleActions[2]
agents[1].NextAction = Settings.PossibleActions[3]

game.Step()
env_1step = game.BuildImage()

fig,ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(env_initial)
ax[1].imshow(env_1step)
fig.set_figheight(15)
fig.set_figwidth(15)
plt.show()
