import os
import sys
import pathlib

# ROOT_DIR = os.path.join(os.getcwd().split("diffuser")[0], "diffuser")

file_dir = os.path.dirname(os.path.abspath(__file__))
# get parent directory
ROOT_DIFFUSERS = pathlib.Path(file_dir).parent
ROOT_MAZE2D = pathlib.Path(file_dir).parent.parent
# ROOT_DIFFUSION_PLANNING = pathlib.Path(file_dir).parent.parent

if __name__ == "__main__":
    # sys.path.append(ROOT_DIR)
    print('ROOT_DIFFUSERS', ROOT_DIFFUSERS)