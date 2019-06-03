import numpy as np
from numpy import *
import matplotlib.pyplot as plt

data = array([np.load("/Users/Lipman/Downloads/model_normalized-7.npy")])[0]

# front = np.sum(data, axis=0)
# side = np.sum(data, axis=1)
# other_size = np.sum(data, axis=2)

#https://github.com/s-macke/VoxelSpace


def capture_picture(re_ax, p, phi_x):

    distance = 64
    screen_width = 128
    screen_height = 128
    data_height = data.shape[1]
    data_width = data.shape[0]
    data_depth = data.shape[2]
    # print("data shape", data.shape)

    #TODO, have this work so we can vary y as well

    sinphi_x = math.sin(phi_x)

    cosphi_x = math.cos(phi_x)

    result = np.zeros((screen_width, screen_height))

    num_found = 0

    # ax = fig.add_subplot(2, 2, 3)
    # Draw from back to the front (high z coordinate to low z coordinate)
    for z in range(distance, p["z"], -1):
        if z >= data_depth:
            continue
        dist_from_point = z - p["z"]

        # print("z: ", z)

        # pleft = {"x": -dist_from_point + p["x"], "z": -dist_from_point + p["z"], "y": -dist_from_point + p["y"]}
        # pright = {"x": dist_from_point + p["x"], "z": -dist_from_point + p["z"], "y": dist_from_point + p["y"]}

        pleft = {"x": (-cosphi_x * dist_from_point - sinphi_x * dist_from_point) + p["x"], "z": (sinphi_x * dist_from_point - cosphi_x * dist_from_point) + p["z"], "y": -dist_from_point + p["y"]}
        pright = {"x": (cosphi_x * dist_from_point - sinphi_x * dist_from_point) + p["x"], "z": (-sinphi_x * dist_from_point - cosphi_x * dist_from_point) + p["z"], "y": dist_from_point + p["y"]}
        # viewpoint_ax.plot([pleft["x"], pright["x"]], [pleft["z"], pright["z"]], color='k', linestyle='-', linewidth=1)
        # ax.plot([pleft["y"], pright["y"]], [pleft["z"], pright["z"]], color='c', linestyle='-', linewidth=1)
        original_x = pleft["x"]
        original_y = pleft["y"]
        # print("pleft: ", pleft, " pright: ", pright)
        # segment the line
        dx = (pright["x"] - pleft["x"]) / screen_width
        dy = (pright["y"] - pleft["y"]) / screen_height
        # print("dx 3d: ", (pright["x"] - pleft["x"]), " dx 2d: ", dx)
        # print("dy 3d: ", (pright["y"] - pleft["y"]), " dy 2d: ", dy)
        # Raster line and draw a vertical line for each segment
        for x in range(0, screen_width):
            if pleft["x"] >= data_width or pleft["x"] <= 0:
                pleft["x"] += dx
                continue
            for y in range(0, screen_height):
                if pleft["y"] >= data_height or pleft["y"] <= 0:
                    pleft["y"] += dy
                    continue
                #TODO, make this vectorized using pytorch so it is differentiable
                if data[int(pleft["x"]), int(pleft["y"]), z] != 0:
                    # print("found one at ", int(pleft["x"]), int(pleft["y"]))
                    num_found += 1
                    result[x, y] = data[int(pleft["x"]), int(pleft["y"]), z]
                    # result[x , int(y / dist_from_point * scale_height)] = data[int(pleft["x"]), int(pleft["y"]), z]
                pleft["y"] += dy
            pleft["y"] = original_y
            pleft["x"] += dx
        pleft["x"] = original_x
    print(num_found)
    re_ax.imshow(result)


num = 30
l = int(sqrt(num))
f, axarr = plt.subplots(l, l)

i = 0
for x in range(l):
    for y in range(l):
        ax1 = axarr[x, y]
        ax1.axis('equal')
        # need to tune these values
        p = {"x": 0, "y": 32, "z": -10}
        phi = 6.2 / (l ** 2) * i
        print(x, y, phi)
        capture_picture(ax1, p, phi)
        i += 1

plt.show()

# f, axarr = plt.subplots(2, 2)
# ax1 = axarr[0, 0]
# ax1.axis('equal')
# p = {"x": 0, "y": 45, "z": 0}
# phi = 0.17222222222222222
# capture_picture(ax1, p, phi)
#
# ax1 = axarr[0, 1]
# ax1.axis('equal')
# p = {"x": 0, "y": 45, "z": 0}
# phi = 0.34444444444444444
# capture_picture(ax1, p, phi)
#
# data = np.swapaxes(data, 1, 2)
#
# ax1 = axarr[1, 0]
# ax1.axis('equal')
# p = {"x": 0, "y": 32, "z": -10}
# phi =5.952
# capture_picture(ax1, p, phi)
#
# ax1 = axarr[1, 1]
# ax1.axis('equal')
# p = {"x": 0, "y": 32, "z": -10}
# phi = 0.6888888888888889
# capture_picture(ax1, p, phi)
#
# plt.show()

#good values of phi, x, y: not really, still nead to mess around
"""
no switch
phi = 0.17222222222222222 p = {"x": 0, "y": 45, "z": 0}
phi = 0.34444444444444444 p = {"x": 0, "y": 45, "z": 0}

np.swapaxes(data, 1, 2)
phi = 5.952 p = {"x": 0, "y": 32, "z": -10}
phi = 0.6888888888888889 p = {"x": 0, "y": 32, "z": -10}
"""
