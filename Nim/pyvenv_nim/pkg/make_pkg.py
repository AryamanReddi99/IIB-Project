# Makes the current folder a usable VSCode package
# To use, place this script at top level of pkg folder, run as administrattor
# Creates __init__.py files recursively
# Symlinks current folder one level up

import os

start_dir = os.path.dirname(os.path.realpath(__file__))

# recursively add __init__.py
for dir, subdirs, files in os.walk(start_dir):
    f = open(os.path.join(dir,"__init__.py"), "w")

# symlink
src = start_dir
src_basename = os.path.basename(start_dir)
dest_dir = os.path.dirname(os.path.dirname(start_dir))
dest = os.path.join(dest_dir,src_basename)
os.symlink(src,dest)

