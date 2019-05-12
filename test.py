from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vals = np.load("/Users/Lipman/Downloads/model_normalized-2.npy")
print(vals.shape[0])
i = 0
for v in np.take(vals, np.random.choice(vals.shape[0], 10000), axis=0):
    i += 1
    if i % 100 == 0:
        print(i)
    ax.scatter(v[0], v[1], v[2], c='r', alpha=.25, marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
