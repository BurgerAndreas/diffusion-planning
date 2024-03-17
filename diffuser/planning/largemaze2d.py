import torch
import numpy as np


# from `diffuser/utils/rendering.py`: 
MAZE_BOUNDS = {
    "maze2d-umaze-v1": (0, 5, 0, 5),
    "maze2d-medium-v1": (0, 8, 0, 8),
    "maze2d-large-v1": (0, 9, 0, 12),
}

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

# Maybe better: construct maze as a proper gym / mujoco env that we can render
# via maze_spec https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/maze_model.py


def global_to_local(global_pos, maze_size, overlap=None, large_maze_outer_wall=False):
    """
    Convert global position (large maze) to local position (small maze).

    Args:
        maze_size: size of the original small maze. without overlap (i.e. the larger numbers)
        large_maze_outer_wall: if True, an outer wall was added to the large maze after removing the overlap

    Example of two small mazes [0,9] with overlap [1,1]: 
        [0,9][9,18] -remove walls-> [1,8][10,17] -virtual overlap-> [0,7][7,14]
        if x > 7 ->
    """
    # fix input types
    global_pos = np.array(global_pos, dtype=float)
    if overlap is not None:
        if np.allclose(overlap, np.array([0, 0])):
            overlap = None
        else:
            overlap = np.array(overlap, dtype=float)

    # remove the outer wall that was added to the large maze
    if large_maze_outer_wall is True:
        global_pos[0] -= 1
        global_pos[1] -= 1

    # add back overlap to the global position
    if overlap is not None:
        small_maze_wo_walls = maze_size - (overlap * 2)
        # which small maze is the global position in?
        maze_coord = np.array([ 
            global_pos[0] / small_maze_wo_walls[0], 
            global_pos[1] / small_maze_wo_walls[1]
            ], dtype=int)
        # print('maze coord', maze_coord, f'from global_pos: {global_pos}')
        global_pos[0] += overlap[0] * 2 * maze_coord[0]
        global_pos[1] += overlap[1] * 2 * maze_coord[1]
        # print('global_pos', global_pos)

    # map global position to local position
    local_pos = np.zeros(2, dtype=int)
    local_pos[0] = global_pos[0] % maze_size[0]
    local_pos[1] = global_pos[1] % maze_size[1]
    # print('local_pos', local_pos)

    # original maze had outer walls
    local_pos[0] += 1
    local_pos[1] += 1

    assert np.all(local_pos >= 0) and np.all(local_pos <= maze_size), \
        f'local_pos: {local_pos} (from global_pos: {global_pos}) and maze_size: {maze_size} and overlap: {overlap}'
    return local_pos

def get_maze_coord_from_global_pos(global_pos, maze_size, overlap=None, large_maze_outer_wall=False):
    """which small maze is the global position in?"""
    # remove the outer wall that was added to the large maze
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


if __name__ == "__main__":
    import diffuser.datasets as datasets
    import diffuser.utils as utils

    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d"

    small_maze = datasets.load_environment(dataset)
    maze_layout = small_maze.maze_arr
    small_maze_size = maze_layout.shape
    print('maze layout\n', maze_layout)
    print('small maze size', small_maze_size)




    # large_maze = generate_large_maze(n_maze_h=2, n_maze_w=2, maze_layout=maze_layout, overlap=overlap)
    # small_maze_size -= (overlap * 2)

    # test 1 - in the first small maze
    overlap = np.array([1, 1])
    pos1 = np.array([0., 0.])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    # need to add one for the removed wall
    assert np.allclose(pos_loc1, np.array([1, 1])), f'pos_loc1: {pos_loc1}'

    overlap = np.array([1, 1])
    pos1 = np.array([1, 1])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap, large_maze_outer_wall=True)
    # need to add one for the removed wall
    assert np.allclose(pos_loc1, np.array([1, 1])), f'pos_loc1: {pos_loc1}'

    overlap = np.array([0, 0])
    pos1 = np.array([1, 1])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([2, 2])), f'pos_loc1: {pos_loc1}'

    overlap = None
    pos1 = np.array([1, 1])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([1, 1])), f'pos_loc1: {pos_loc1}'

    # test 2 - one inside one outside the first small maze
    overlap = np.array([1, 1])
    pos1 = np.array([10, 9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([3, 10])), f'pos_loc1: {pos_loc1}'

    overlap = np.array([0, 0])
    pos1 = np.array([10, 9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([1, 9])), f'pos_loc1: {pos_loc1}'

    overlap = None
    pos1 = np.array([10, 9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([1, 9])), f'pos_loc1: {pos_loc1}'

    # test 3 - just inside the first small maze
    overlap = np.array([1, 1])
    pos1 = np.array([7.9, 10.9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    # need to add one for the removed wall
    assert np.allclose(pos_loc1, np.array([9.9, 12.9])), f'pos_loc1: {pos_loc1}'

    overlap = np.array([0, 0])
    pos1 = np.array([8.9, 11.9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, pos1), f'pos_loc1: {pos_loc1}'

    overlap = None
    pos1 = np.array([8.9, 11.9])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, pos1), f'pos_loc1: {pos_loc1}'

    # test 4 - just outside the first small maze
    overlap = np.array([1, 1])
    pos1 = np.array([8.1, 11.1])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    # need to add two for the removed walls
    assert np.allclose(pos_loc1, np.array([1.1, 1.1])), f'pos_loc1: {pos_loc1}'

    overlap = np.array([0, 0])
    pos1 = np.array([9.1, 12.1]) # 9 -> 0
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([.1, .1])), f'pos_loc1: {pos_loc1}'

    overlap = None
    pos1 = np.array([9.1, 12.1])
    pos_loc1 = global_to_local(pos1, small_maze_size, overlap)
    assert np.allclose(pos_loc1, np.array([.1, .1])), f'pos_loc1: {pos_loc1}'

    print('all tests passed!')