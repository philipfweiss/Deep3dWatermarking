import numpy as np
from numpy import *
import matplotlib.pyplot as plt

data = array([np.load("/Users/Lipman/Downloads/model_normalized-7.npy")])[0]

front = np.sum(data, axis=0)
side = np.sum(data, axis=1)
other_size = np.sum(data, axis=2)

fig = plt.figure()

ax = fig.add_subplot(1, 4, 1)
ax.imshow(front)

ax = fig.add_subplot(1, 4, 2)
ax.imshow(side)

ax = fig.add_subplot(1, 4, 3)
ax.imshow(other_size)

#https://github.com/s-macke/VoxelSpace

p = {"x": 32, "y": 32, "z": -30}
height = 0
horizon_y = 30
horizon_x = 30
scale_height = 2
distance = 50
screen_width = 32
screen_height = 32
data_height = data.shape[1]
data_width = data.shape[0]
data_depth = data.shape[2]
print("data shape", data.shape)

result = np.zeros((screen_width, screen_height))

num_found = 0

# Draw from back to the front (high z coordinate to low z coordinate)
for z in range(distance, p["z"], -1):
    if z >= data_depth:
        continue
    print("z: ", z)
    # Find line on map. This calculation corresponds to a field of view of 90°
    pleft = {"x": -z + p["x"], "y": z + p["y"]}
    pright = {"x": z + p["x"], "y": -z + p["y"]}
    original_x = pleft["x"]
    original_y = pleft["y"]
    print("pleft: ", pleft, " pright: ", pright)
    # segment the line
    dx = (pright["x"] - pleft["x"]) / screen_width
    dy = (pright["y"] - pleft["y"]) / screen_height
    print("dx 3d: ", (pright["x"] - pleft["x"]), " dx 2d: ", dx)
    print("dy 3d: ", (pright["y"] - pleft["y"]), " dy 2d: ", dy)
    # Raster line and draw a vertical line for each segment
    for x in range(int(pleft["x"] / data_width * screen_width), int(pright["x"] / data_width * screen_width)):
        if pleft["x"] >= data_width or pleft["x"] <= 0:
            pleft["x"] += dx
            continue
        for y in range(int(pright["y"] / data_height * screen_height) - 1, int(pleft["y"] / data_height * screen_height), 1):
            if pleft["y"] >= data_height or pleft["y"] <= 0:
                pleft["y"] += dy
                continue
            print(x, y)
            if data[int(pleft["x"]), int(pleft["y"]), z] != 0:
                num_found += 1
                print("drawing from: ", int(pleft["x"]), int(pleft["y"]), z, " to assign at: ", x, y)
                result[int(x / z * scale_height) + horizon_x, int(y / z * scale_height) + horizon_y] = data[int(pleft["x"]), int(pleft["y"]), z]
            pleft["y"] += dy
        pleft["y"] = original_y
        pleft["x"] += dx
    pleft["x"] = original_x

print(num_found)
ax = fig.add_subplot(1, 4, 4)
ax.imshow(result)

plt.show()


        # height_on_screen = (height - heightmap[pleft.x, pleft.y]) / z * scale_height. + horizon
        # DrawVerticalLine(i, height_on_screen, screen_height, colormap[pleft.x, pleft.y])

