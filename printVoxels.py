import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D



def draw_voxels(data, ax):
    front = np.sum(data, axis=1)
    return ax.imshow(front)

    # data[data != 0] = 1
    # filled = data
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.grid(True)
    # ax.view_init(30, 145)
    # ax.voxels(filled, facecolors='#1f77b430', edgecolors='#0101FD20')

