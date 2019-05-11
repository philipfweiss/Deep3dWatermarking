import os
import shutil


shape_core_dir = "/home/jlipman500/ShapeNetCore.v2"
new_files_dir = "/home/jlipman500/ShapeNetCore.slice"
old_files_dir_name = "old_files"

objects = os.listdir(new_files_dir)

old_files_dir_path = os.path.join(shape_core_dir, old_files_dir_name)

if not os.path.isdir(old_files_dir_path):
    os.makedirs(old_files_dir_path)


for object in objects:
    shutil.move(os.path.join(new_files_dir, object), os.path.join(old_files_dir_path, object))

shutil.rmtree(new_files_dir)
