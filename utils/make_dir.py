import os

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
