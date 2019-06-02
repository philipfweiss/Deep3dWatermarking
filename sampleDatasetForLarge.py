import os
import shutil
import json
import argparse
import pymesh
from pyntcloud import PyntCloud
import numpy as np

parser = argparse.ArgumentParser(description='Only take shape files with a certain complexity.')
parser.add_argument('--cutoff', type=int, help='an integer for the min number of vertices to cut off at')
parser.add_argument('--size', type=int, help='an integer for the max number of examples to pick', default=500)
parser.add_argument('--norecompute', help='if program should recompute numpy files')
parser.add_argument('--pcsize', type=int, help='the density of the point cloud')
parser.add_argument('--voxelspd', type=int, help='the number of voxels per dimension')

args = parser.parse_args()

shape_core_dir = "/home/jlipman500/ShapeNetCore.v2"
new_files_dir = "/home/jlipman500/ShapeNetCore.slice"
obj_file_format = "model_normalized.obj"
np_file_format = "model_normalized.npy"
ply_file_format = "model_normalized.ply"

vertice_cutoff = 10000
if "--cutoff" in args:
    vertice_cutoff = args["--cutoff"]

total_slice_size = args["--size"]

recompute = True
if "--norecompute" in args:
    recompute = False

point_cloud_size = 100000
if "--pcsize" in args:
    point_cloud_size = args["--pcsize"]

voxels_per_dim = 64
if "--voxelspd" in args:
    voxels_per_dim = args["--voxelspd"]

if not os.path.isdir(new_files_dir):
    os.makedirs(new_files_dir)

groups = os.listdir(shape_core_dir)

def save_as_ply(filename, new_filename):
    mesh = pymesh.load_mesh(filename)
    vertices = mesh.vertices
    faces = mesh.faces
    pymesh.save_mesh_raw(new_filename, vertices, faces)

def convert_to_np_array(filename, new_filename):
    anky = PyntCloud.from_file(filename)
    anky_cloud = anky.get_sample("mesh_random", n=point_cloud_size, rgb=False, normals=False, as_PyntCloud=True)
    voxelgrid_id = anky_cloud.add_structure("voxelgrid", n_x=voxels_per_dim, n_y=voxels_per_dim, n_z=voxels_per_dim)

    voxelgrid = anky_cloud.structures[voxelgrid_id]

    binary_feature_vector = voxelgrid.get_feature_vector(mode="density")
    np.save(new_filename, binary_feature_vector)

def move_files(vertice_cutoff, total_slice_size):
    size_of_slice = 0
    total_number = 0
    errors = 0
    for group in groups:
        objects = os.listdir(os.path.join(shape_core_dir, group))
        for object in objects:
            try:
                with open(os.path.join(shape_core_dir, group, object, "models", "model_normalized.json")) as metadata:
                    data = json.load(metadata)
                    total_number += 1
                    if data["numVertices"] > vertice_cutoff:
                        shutil.move(os.path.join(shape_core_dir, group, object), os.path.join(new_files_dir, object))
                        model_dir = os.path.join(new_files_dir, object, "models")
                        np_file_name = os.path.join(model_dir, np_file_format)
                        obj_file_name = os.path.join(model_dir, obj_file_format)
                        ply_file_name = os.path.join(model_dir, ply_file_format)
                        if recompute or (not os.path.isfile(np_file_name) or not os.path.isfile(ply_file_name)):
                            if os.path.isfile(np_file_name):
                                os.remove(np_file_name)
                            if os.path.isfile(ply_file_name):
                                os.remove(ply_file_name)
                            save_as_ply(obj_file_name, ply_file_name)
                            convert_to_np_array(ply_file_name, np_file_name)
                        size_of_slice += 1
                        if total_slice_size <= size_of_slice:
                            return (size_of_slice, total_number, errors)
            except Exception as e:
                print(e)
                errors += 1
        print("group: ", group, " slice size: ", size_of_slice, " total number: ", total_number)
    return (size_of_slice, total_number, errors)

size_of_slice, total_number, errors = move_files(vertice_cutoff, total_slice_size)
print("Completed. Final stats: slice size: ", size_of_slice, " total number: ", total_number, " num error: ", errors)
