from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vals = np.load("/Users/Lipman/Downloads/model_normalized-3.npy")
print(vals.shape)

i = 0

for x, xidx in enumerate(vals):
    for y, yidx in enumerate(x):
        for z, zidx in enumerate(y):
            i += 1
            if i % 100 == 0:
                print(i)
            if z == 1.0:
                ax.scatter(xidx, yidx, zidx, c='r', alpha=.25, marker='.', depthshade=True)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
