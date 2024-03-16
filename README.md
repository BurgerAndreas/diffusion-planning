# diffusion-planning
Planning in large domains with continuous diffusion and discrete planners - Andreas Burger, Jack Sun, Yifan Ruan

Go to `scripts/run_diffusion_planner.py`! (Work in progress)

### Installation
```bash
conda env create -f environment.yml # this will partly fail, but it will create the environment
conda activate diffuser
pip install -e .

pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
pip install -r requirements.txt 

# check your cuda version with nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# on Ubuntu 22.04
# https://github.com/openai/mujoco-py/issues/492#issuecomment-607688770
sudo apt-get install patchelf
sudo apt-get install libglu1-mesa-dev mesa-common-dev
# install old gcc version. Ideally gcc-7
gcc --version # check your gcc version
sudo apt-get install build-essential
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --config gcc # select gcc-9
pip install mujoco_py==2.0.2.8
pip install "cython<3"
sudo update-alternatives --config gcc # select whatever gcc version you had before

# ./scripts/download_pretrained.sh
```

### Run


Original diffuser:
```bash
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```

### Maze2d

In this repo
- Using the env: `scripts/plan_maze2d.py`
- Loading the env from d4rl: `diffuser/datasets/d4rl.py`
- Rendering, maze size: `diffuser/utils/rendering.py`

Outside this repo
- [Documentation](https://robotics.farama.org/envs/maze/point_maze/)
- [MazeEnv source code](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/maze_model.py)
- [How to create your own point maze env](https://github.com/Farama-Foundation/Minari/blob/fa9b2e8ed4bad7f2010819709ce74f0400a4acac/docs/tutorials/dataset_creation/point_maze_dataset.py#L6)
- [How dataset was created](https://github.com/Farama-Foundation/D4RL/blob/master/scripts/generation/generate_maze2d_datasets.py)

### Resources

Maze2d

Based on 
- https://github.com/jannerm/diffuser/tree/maze2d
- https://github.com/huggingface/diffusers/tree/main/examples/reinforcement_learning