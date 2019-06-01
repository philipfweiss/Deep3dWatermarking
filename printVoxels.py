import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D



def draw_voxels(data, ax):
    data[data != 0] = 1
    filled = data[0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    ax.view_init(30, 145)
    ax.voxels(filled, facecolors='#1f77b430', edgecolors='#0101FD20')
data = array([np.load("/Users/Lipman/Downloads/model_normalized-7.npy")])
fig = plt.figure()
draw_voxels(data, fig)
plt.show()

