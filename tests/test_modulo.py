import numpy as np

global_pos = np.array([14, 43])

small_maze_wo_walls = np.array([7, 10])


local_pos = np.zeros(2, dtype=float)
local_pos[0] = global_pos[0] % small_maze_wo_walls[0]
local_pos[1] = global_pos[1] % small_maze_wo_walls[1]

print(f'Local position: {local_pos}')



maze_coord = [global_pos[0] // small_maze_wo_walls[0], global_pos[1] // small_maze_wo_walls[1]]
print(f'Maze coordinates: {maze_coord}')

global_pos[0] = local_pos[0] + (small_maze_wo_walls[0] * maze_coord[0])
global_pos[1] = local_pos[1] + (small_maze_wo_walls[1] * maze_coord[1])

print(f'Global position: {global_pos}')