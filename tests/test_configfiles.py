import pickle
import os
import sys

file_path = os.path.dirname(__file__)
parent_path = os.path.join(file_path, "..")
path = "logs/maze2d-large-v1/diffusion/H384_T256/render_config.pkl"
full_path = os.path.join(parent_path, path)

with open(full_path, 'rb') as f:
    data = pickle.load(f)
    print('data: ', data)