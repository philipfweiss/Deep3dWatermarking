import os
import shutil
import json
import argparse
from pyntcloud import PyntCloud
import numpy as np

parser = argparse.ArgumentParser(description='Only take shape files with a certain complexity.')
parser.add_argument('--cutoff', type=int, help='an integer for the min number of vertices to cut off at')
parser.add_argument('--size', type=int, help='an integer for the max number of examples to pick')
parser.add_argument('--norecompute', help='if program should recompute numpy files')

args = parser.parse_args()

shape_core_dir = "/home/jlipman500/ShapeNetCore.v2"
new_files_dir = "/home/jlipman500/ShapeNetCore.slice"
obj_file_format = "model_normalized.obj"
np_file_format = "model_normalized.npy"

vertice_cutoff = 10000
if "--cutoff" in args:
    vertice_cutoff = args["--cutoff"]

total_slice_size = 500
if "--size" in args:
    total_slice_size = args["--size"]

recompute = True
if "--norecompute" in args:
    recompute = False

if not os.path.isdir(new_files_dir):
    os.makedirs(new_files_dir)

groups = os.listdir(shape_core_dir)

def convert_to_np_array(filename, new_filename):
    anky = PyntCloud.from_file(filename)
    anky_cloud = anky.get_sample("mesh_random", n=100000, rgb=False, normals=False, as_PyntCloud=True)
    print(anky_cloud)
    print(anky_cloud.points.values)
    np.save(new_filename, anky_cloud.points.values)

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
                        new_file_name = os.path.join(model_dir, np_file_format)
                        obj_file_name = os.path.join(model_dir, obj_file_format)
                        if recompute and os.path.isfile(new_file_name):
                            os.remove(new_file_name)
                            convert_to_np_array(obj_file_name, new_file_name)
                        elif not os.path.isfile(new_file_name):
                            convert_to_np_array(obj_file_name, new_file_name)
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
