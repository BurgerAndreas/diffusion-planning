import torch
import numpy as np

import gym
import d4rl

from typing import Tuple, List, Dict, Any, Union, Optional, Sequence, Iterable

from contextlib import contextmanager
import sys, os
import copy

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# from `diffuser/utils/rendering.py`: 
MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
}

# maze_map format
WALL = 10
EMPTY = 11
GOAL = 12

# maze_spec format
# from /home/<user>/miniforge3/envs/diffuser/lib/python3.8/site-packages/d4rl/pointmaze/maze_model.py
U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

map_to_spec = {
    10: '#',
    11: 'O',
    12: 'G',
}


# TODO(Yifan, Jack): make this to whatever we need for the planner
# just some ideas for now
def generate_large_maze(maze_layout, n_maze_h=2, n_maze_w=2, overlap=None, large_maze_outer_wall=False):
    """
    Concatenate multiple smaller mazes to create a larger maze
    """
    # each maze is surrounded by walls, so remove them (10 -> 11)
    maze_layout[0, :] = 11
    maze_layout[-1, :] = 11
    maze_layout[:, 0] = 11
    maze_layout[:, -1] = 11
    if overlap is not None:
        # remove rows and columns from the smaller maze
        maze_layout = maze_layout[overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
    # remove random goal (12)
    maze_layout[maze_layout == 12] = 11
    large_maze = np.tile(maze_layout, (n_maze_h, n_maze_w))
    if overlap is not None:
        # extend the large maze by 1 row and column
        large_maze = np.pad(large_maze, ((1, 1), (1, 1)), mode='constant', constant_values=11)
    if large_maze_outer_wall == True:
        # add walls around the large maze
        large_maze[0, :] = 10
        large_maze[-1, :] = 10
        large_maze[:, 0] = 10
        large_maze[:, -1] = 10
    return large_maze

def parse_maze(maze_str):
    """Maze spec to maze map."""
    # from /home/<user>/miniforge3/envs/diffuser/lib/python3.8/site-packages/d4rl/pointmaze/maze_model.py
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr

def maze_map_to_maze_spec(maze_map: Union[np.ndarray, List[List[int]]]) -> str:
    """Convert maze map (10, 11, 12) to maze spec (str)."""
    maze_spec = ""
    for row in maze_map:
        maze_spec += ''.join([map_to_spec[r] for r in row])
        maze_spec += '\\'
    # remove trailing backslash
    maze_spec = maze_spec[:-1]
    return maze_spec

def maze_to_gym_env(maze):
    """Convert maze as a proper gym / mujoco env that we can render.
    Via maze_spec https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/maze_model.py
    /home/<user>/miniforge3/envs/diffuser/lib/python3.8/site-packages/d4rl/pointmaze/maze_model.py
    """
    maze_spec = maze_map_to_maze_spec(maze)
    env = d4rl.pointmaze.maze_model.MazeEnv(maze_spec=maze_spec)
    # gym.envs.registration.register(
    #     id='gym_examples/MazeEnv-v0',
    #     # entry_point='gym_examples.envs:GridWorldEnv',
    #     entry_point=d4rl.pointmaze.maze_model.MazeEnv,
    #     max_episode_steps=300,
    # )
    return env

def global_to_local(_global_pos, _maze_size, overlap, large_maze_outer_wall):
    """Convert global position (large maze) to local position (small maze).

    Args:
        maze_size: size of the original small maze. without overlap (i.e. the larger numbers)
        large_maze_outer_wall: if True, an outer wall was added to the large maze after removing the overlap

    Example of two small mazes [0,9] with overlap [1,1]: 
        [0,9][9,18] -remove walls-> [1,8][10,17] -virtual overlap-> [0,7][7,14]
        if x > 7 ->
    """
    # fix input types
    global_pos = np.array(_global_pos, dtype=float)
    maze_size = np.array(_maze_size, dtype=float)
    if overlap is not None:
        if np.allclose(overlap, np.array([0, 0])):
            overlap = None
        else:
            overlap = np.array(overlap, dtype=float)

    # remove the outer wall that was added to the large maze
    if large_maze_outer_wall is True:
        global_pos[0] -= 1
        global_pos[1] -= 1

    # map global position to local position (to unit maze size)
    local_pos = np.zeros(2, dtype=float)
    if overlap is not None:
        small_maze_wo_walls = maze_size - (overlap * 2)
        local_pos[0] = global_pos[0] % small_maze_wo_walls[0]
        local_pos[1] = global_pos[1] % small_maze_wo_walls[1]
    else:
        local_pos[0] = global_pos[0] % maze_size[0]
        local_pos[1] = global_pos[1] % maze_size[1]

    # add back overlap
    if overlap is not None:
        # which small maze is the global position in?
        # small_maze_wo_walls = maze_size - (overlap * 2)
        # maze_coord = np.array([ 
        #     global_pos[0] / small_maze_wo_walls[0], 
        #     global_pos[1] / small_maze_wo_walls[1]
        #     ])
        # maze_coord = np.round(maze_coord).astype(float)
        # add overlap for the removed outer walls
        local_pos[0] += overlap[0]
        local_pos[1] += overlap[1]
    
    # print('  local_pos =', local_pos, '| global_pos =', global_pos, '| maze_size =', maze_size, '| overlap =', overlap, '| large_maze_outer_wall =', large_maze_outer_wall)
    # print('  global_pos[0] % maze_size[0] =', f'{global_pos[0]} % {maze_size[0]} =', global_pos[0] % maze_size[0])
    # print('  global_pos[1] % maze_size[1] =', f'{global_pos[1]} % {maze_size[1]} =', global_pos[1] % maze_size[1])

    assert np.all(local_pos >= 0) and np.all(local_pos <= maze_size), \
        f'local_pos: {local_pos} (from global_pos: {global_pos}) and maze_size: {maze_size} and overlap: {overlap}'
    return local_pos

def local_to_global(_local_pos, maze_coord, _maze_size, overlap, large_maze_outer_wall):
    """Convert local position (small maze) to global position (large maze).
    Reverse of `global_to_local`.

    Args:
        maze_coord nd.array[int,int]: which small maze is the local position in?
        maze_size: size of the original small maze. without overlap (i.e. the larger numbers)
        large_maze_outer_wall: if True, an outer wall was added to the large maze after removing the overlap
    """
    # fix input types
    local_pos = np.array(_local_pos, dtype=float)
    maze_size = np.array(_maze_size, dtype=float)
    if overlap is not None:
        if np.allclose(overlap, np.array([0, 0])):
            overlap = None
        else:
            overlap = np.array(overlap, dtype=float)
    
    # remove overlap from the local position
    if overlap is not None:
        local_pos[0] -= overlap[0]
        local_pos[1] -= overlap[1]
    
    # map local position to global position 
    global_pos = np.zeros(2, dtype=float)
    if overlap is not None:
        small_maze_wo_walls = maze_size - (overlap * 2)
        global_pos[0] = local_pos[0] + (small_maze_wo_walls[0] * maze_coord[0])
        global_pos[1] = local_pos[1] + (small_maze_wo_walls[1] * maze_coord[1])
    else:
        global_pos[0] = local_pos[0] + (maze_size[0] * maze_coord[0])
        global_pos[1] = local_pos[1] + (maze_size[1] * maze_coord[1])

    # add outer wall
    if large_maze_outer_wall is True:
        global_pos[0] += 1
        global_pos[1] += 1

    return global_pos

def global_to_local_openmaze(global_traj1, global_traj2, open_maze_size):
    """Convert global positions (large maze) to local positions (small maze).
    Takes in pieces of trajectories.
    Local frame tries to put the midpoint between the two global trajectories in the center of the open maze.

    Args:
    - global_traj1: [global_pos1, ..., global_posN]

    Usage:
        ltraj1, ltraj2, ltog_offset = global_to_local_openmaze(gtraj1, gtraj2, open_maze_size)
        ltraj1 == gtraj1 - ltog # True
        gtraj1 == ltraj1 + ltog # True
    """
    # just take the enpoints for now
    if global_traj1.ndim > 1:
        global_pos1 = global_traj1[-1]
    else:
        global_pos1 = global_traj1
    if global_traj2.ndim > 1:
        global_pos2 = global_traj2[0]
    else:
        global_pos2 = global_traj2

    # # find the midpoint (in global frame)
    # # mid_g = (global_pos1 + global_pos2) / 2
    # mid_g = np.zeros_like(global_pos1) 
    # mid_g[0] = (global_pos1[0] + global_pos2[0]) / 2
    # mid_g[1] = (global_pos1[1] + global_pos2[1]) / 2
    # # enforce that the midpoint is in the center of the open maze
    # open_maze_center = np.asarray(open_maze_size) / 2
    # gtol = open_maze_center - mid_g
    # # convert global pos to local
    # local_pos1 = global_pos1 + gtol
    # local_pos2 = global_pos2 + gtol
    # # print(f'global_pos1 = {global_pos1} | global_pos2 = {global_pos2}')
    # # print(f'midpoint = {mid_g} | gtol = {gtol} | local_pos1 = {local_pos1} | local_pos2 = {local_pos2}')
    # # convert whole trajectory to local
    # local_traj1 = global_traj1 + gtol
    # local_traj2 = global_traj2 + gtol
    # # local to global: glob = local + ltog
    # ltog = -gtol
    # return local_traj1, local_traj2, ltog
        
    gap = global_pos2 - global_pos1
    open_maze_center = np.asarray(open_maze_size) / 2
    gtol = - global_pos1 + open_maze_center - gap / 2
    # convert whole trajectory to local
    local_traj1 = global_traj1 + gtol
    local_traj2 = global_traj2 + gtol
    # local to global: glob = local + ltog
    ltog = -gtol
    return local_traj1, local_traj2, ltog



def get_maze_coord_from_global_pos(_global_pos, maze_size, overlap, large_maze_outer_wall):
    """Which small maze is the global position in?"""
    # remove the outer wall that was added to the large maze
    global_pos = copy.deepcopy(_global_pos)
    if large_maze_outer_wall is True:
        global_pos[0] -= 1
        global_pos[1] -= 1
    # add back overlap to the global position
    if overlap is not None:
        small_maze_wo_walls = maze_size - (overlap * 2)
    else:
        small_maze_wo_walls = maze_size
    maze_coord = np.array([ 
        global_pos[0] / small_maze_wo_walls[0], 
        global_pos[1] / small_maze_wo_walls[1]
        ], dtype=int)
    return maze_coord



def get_start_goal(global_start, global_goal, large_maze_size):
    top = 1.5
    bottom = large_maze_size[0] - 1.5
    middle = large_maze_size[0] / 2
    left = 1.5
    right = large_maze_size[1] - 1.5
    
    if type(global_start) is str:
        global_start = np.array([
            eval(global_start.split('_')[0]),
            eval(global_start.split('_')[1]),
        ])
    else:
        global_start = np.array(global_start)

    if type(global_goal) is str:
        global_goal = np.array([
            eval(global_goal.split('_')[0]),
            eval(global_goal.split('_')[1]),
        ])
    else:
        global_goal = np.array(global_goal)
    
    return global_start, global_goal

if __name__ == "__main__":
    import diffuser.datasets as datasets
    import diffuser.utils as utils

    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d"

    # d4rl.pointmaze.maze_model.MazeEnv
    small_maze = datasets.load_environment(dataset)
    print(f'small_maze: {small_maze} {type(small_maze)}')
    maze_layout = small_maze.maze_arr
    small_maze_size = maze_layout.shape
    print('maze layout\n', maze_layout)
    print('small maze size', small_maze_size)


    # test 1 - in the first small maze
    overlap = np.array([1, 1])
    pos1 = np.array([0., 0.])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    # need to add one for the removed wall
    assert np.allclose(pos_loc1, np.array([1, 1])), f'pos_loc1: {pos_loc1}'

    ######################################################################

    local_pos = np.array([1, 1])
    maze_coord = np.array([0, 0])
    maze_size = np.array([9, 12])
    gpos = local_to_global(local_pos, maze_coord, maze_size, overlap=None, large_maze_outer_wall=False)
    assert np.allclose(gpos, np.array([1, 1])), f'gpos: {gpos}'

    local_pos = np.array([1, 1])
    maze_coord = np.array([1, 1])
    maze_size = np.array([9, 12])
    gpos = local_to_global(local_pos, maze_coord, maze_size, overlap=None, large_maze_outer_wall=False)
    assert np.allclose(gpos, np.array([10, 13])), f'gpos: {gpos}'

    local_pos = np.array([1, 1])
    maze_coord = np.array([0, 0])
    maze_size = np.array([9, 12])
    gpos = local_to_global(local_pos, maze_coord, maze_size, overlap=overlap, large_maze_outer_wall=False)
    assert np.allclose(gpos, np.array([0, 0])), f'gpos: {gpos}'

    ######################################################################
    # try to build large maze gym env
    # https://robotics.farama.org/envs/maze/point_maze/#custom-maze

    # print registered envs
    all_envs = gym.envs.registry.all()
    maze_envs = [env_spec for env_spec in all_envs if 'maze' in env_spec.id.lower()]
    # print(f'gym envs: {maze_envs}')
    # print(f'gym envs: {gym.envs.registry.keys()}')


    # none of these work
    # example_map = [
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 1, 1, 1, 1]
    # ]
    # env = gym.make('PointMaze_UMaze-v3', maze_map=example_map)
    # env = gym.make('maze2d-large-v1', maze_map=example_map)
    # env = gym.make('maze2d-v1', maze_map=example_map)
    # env = gym.make('MazeEnv', maze_map=example_map)
    # env = gym.make('MazeEnv-v0', maze_map=example_map)

    # this works
    large_maze = generate_large_maze(n_maze_h=2, n_maze_w=2, maze_layout=maze_layout, overlap=overlap)
    # small_maze_size -= (overlap * 2)

    maze_spec = maze_map_to_maze_spec(large_maze)
    # print('umaze', U_MAZE)
    # print('large maze spec\n', maze_spec)
    env = d4rl.pointmaze.maze_model.MazeEnv(maze_spec=maze_spec)

    # gym.envs.registration.register(
    #     id='gym_examples/MazeEnv-v0',
    #     # entry_point='gym_examples.envs:GridWorldEnv',
    #     entry_point=d4rl.pointmaze.maze_model.MazeEnv,
    #     max_episode_steps=300,
    # )
