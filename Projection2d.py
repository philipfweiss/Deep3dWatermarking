import numpy as np
from numpy import *
import torch
import matplotlib.pyplot as plt


# front = np.sum(data, axis=0)
# side = np.sum(data, axis=1)
# other_size = np.sum(data, axis=2)

# https://github.com/s-macke/VoxelSpace for all equations



def capture_picture(p, data, output_height, output_width):
    distance = 64
    screen_width = output_width
    screen_height = output_height
    data_height = data.shape[1]
    data_width = data.shape[0]
    data_depth = data.shape[2]
    # print("data shape", data.shape)

    phi_x = p["phi"]
    sinphi_x = math.sin(phi_x)

    cosphi_x = math.cos(phi_x)

    result = np.zeros((screen_width, screen_height))

    num_found = 0

    dz = -8
    ddz = 0.3
    # a = z_dif * dz_end * ddz ** 2
    # dz = a
    # print(a)
    # Draw from back to the front (high z coordinate to low z coordinate)
    z = distance
    while z > p["z"]:
        # print(z)
        if z >= data_depth:
            z += dz
            dz += ddz
            if dz >= -1:
                dz = -1
            continue
        dist_from_point = z - p["z"]

        pleft = {"x": (-cosphi_x * dist_from_point - sinphi_x * dist_from_point) + p["x"],
                 "z": (sinphi_x * dist_from_point - cosphi_x * dist_from_point) + p["z"],
                 "y": -dist_from_point + p["y"]}
        pright = {"x": (cosphi_x * dist_from_point - sinphi_x * dist_from_point) + p["x"],
                  "z": (-sinphi_x * dist_from_point - cosphi_x * dist_from_point) + p["z"],
                  "y": dist_from_point + p["y"]}

        original_x = pleft["x"]
        original_y = pleft["y"]
        # segment the line
        dx = (pright["x"] - pleft["x"]) / screen_width
        dy = (pright["y"] - pleft["y"]) / screen_height
        # Raster line and draw a vertical line for each segment

        for x in range(0, screen_width):
            if pleft["x"] >= data_width or pleft["x"] <= 0:
                pleft["x"] += dx
                continue
            for y in range(0, screen_height):
                if pleft["y"] >= data_height or pleft["y"] <= 0:
                    pleft["y"] += dy
                    continue
                if data[int(pleft["x"]), int(pleft["y"]), int(z)] != 0:
                    num_found += 1
                    result[x, y] = data[int(pleft["x"]), int(pleft["y"]), int(z)]
                pleft["y"] += dy
            pleft["y"] = original_y
            pleft["x"] += dx
        pleft["x"] = original_x
        z += dz
        dz += ddz
        if dz >= -1:
            dz = -1
    print(num_found)
    return result


ps = [{"x": 16, "y": 40, "z": -10, "phi": 3.875}, {"x": 16, "y": 40, "z": 0, "phi": 5.425},
      dict(x=10, y=32, z=0, phi=3.875),
      {"x": 10, "y": 32, "z": 0, "phi": 5.425}, {"x": 32, "y": 40, "z": -10, "phi": 0},
      {"x": 32, "y": 40, "z": -10, "phi": 3.1}]


def convert_to_2d(data):
    # N x 1 x 64 x 64 x 64 --> N x 6 x 128 x 128
    results = []
    for image in data:
        print("converted image")
        image = image[0]
        image_results = []

        # image_results.append(capture_picture(ps[0], image, 64, 64))
        # image_results.append(capture_picture(ps[1], image, 64, 64))
        #
        # image = image.permute(0, 2, 1)
        #
        # image_results.append(capture_picture(ps[2], image, 64, 64))
        # image_results.append(capture_picture(ps[3], image, 64, 64))
        #
        # image = image.permute(0, 2, 1)
        image = image.permute(2, 1, 0)

        image_results.append(capture_picture(ps[4], image, 64, 64))
        image_results.append(capture_picture(ps[5], image, 64, 64))
        results.append(image_results)

    pytens = torch.tensor(results)
    return pytens

# data = array([np.load("/Users/Lipman/Downloads/model_normalized-7.npy")])[0]
# convert_to_2d(data)

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
#
# plt.subplots(l, l)

# good values of phi, x, y: not really, still nead to mess around
"""
no switch
phi 3.875 x 16 "y": 40, "z": -10}
phi 5.425 x 16 "y": 40, "z": 0}

data = np.swapaxes(data, 1, 2)
phi 3.875 z 0 {"x": 10, "y": 32}
phi 5.425 z 0 {"x": 10, "y": 32}

data = np.swapaxes(data, 1, 2)
data = np.swapaxes(data, 0, 2)
phi 0 x 32 "y": 40, "z": -10}
phi 3.1 x 32 "y": 40, "z": -10}
"""
