from pyntcloud import PyntCloud
import pymesh
import numpy as np


def mesh_to_point_cloud(filename):
    anky = PyntCloud.from_file(filename)
    #anky = pymesh.load_mesh(filename)
    print(anky)
    anky_point_cloud = anky.get_sample("mesh_random", n=10000, rgb=False, normals=False)
    print(anky_point_cloud)

def point_cloud_to_mesh(pcl):
    pass

def save_as_ply(filename, new_filename):
    mesh = pymesh.load_mesh(filename)
    vertices = mesh.vertices
    faces = mesh.faces
    pymesh.save_mesh_raw(new_filename, vertices, faces)


def convert_to_np_array(filename, new_filename):
    mesh = pymesh.load_mesh(filename)
    vertices = mesh.vertices
    np.save(new_filename, vertices)



if __name__ == "__main__":
    test_filename = "data/test_model/model_normalized.obj"
    np_filename = "data/test_model/model_normalized.npy"
    ply_filename = "data/test_model/model_normalized.ply"
    #convert_to_np_array(test_filename, out_filename)
    #save_as_ply(test_filename, ply_filename)
    mesh_to_point_cloud(ply_filename)
