import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D



def draw_voxels(data, fig):
    def make_ax(grid=False):
        ax = fig.gca(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(grid)
        return ax

    data[data != 0] = 1
    filled = data[0]

    ax = make_ax(True)
    ax.view_init(30, 145)
    ax.voxels(filled, facecolors='#1f77b430', edgecolors='#0101FD20')
data = array([np.load("/Users/Lipman/Downloads/model_normalized-7.npy")])
fig = plt.figure()
draw_voxels(data, fig)
plt.show()

