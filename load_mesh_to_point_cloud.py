import pymesh
import numpy as np


def convert_to_np_array(filename, new_filename):
    mesh = pymesh.load_mesh(filename)
    vertices = mesh.vertices
    np.save(new_filename, vertices)

if __name__ == "__main__":
    test_filename = "data/test_model/model_normalized.obj"
    out_filename = "data/test_model/model_normalized.npy"
    convert_to_np_array(test_filename, out_filename)
