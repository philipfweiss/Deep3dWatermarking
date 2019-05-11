import os
import shutil
import json
import argparse

parser = argparse.ArgumentParser(description='Only take shape files with a certain complexity.')
parser.add_argument('--cutoff', type=int, help='an integer for the min number of vertices to cut off at')

args = parser.parse_args()

shape_core_dir = "/home/jlipman500/ShapeNetCore.v2"
new_files_dir = "/home/jlipman500/ShapeNetCore.slice"
vertice_cutoff = 10000
if "--cutoff" in args:
    vertice_cutoff = args["--cutoff"]
size_of_slice = 0
total_number = 0
errors = 0

# shutil.rmtree(new_files_dir)
os.makedirs(new_files_dir)

groups = os.listdir(shape_core_dir)

for group in groups:
    objects = os.listdir(os.path.join(shape_core_dir, group))
    for object in objects:
        try:
            with open(os.path.join(shape_core_dir, group, object, "models", "model_normalized.json")) as metadata:
                data = json.load(metadata)
                total_number += 1
                if data["numVertices"] > vertice_cutoff:
                    shutil.move(os.path.join(shape_core_dir, group, object), os.path.join(new_files_dir, object))
                    size_of_slice += 1
        except:
            errors += 1
    print("slice size: ", size_of_slice, " total number: ", total_number)