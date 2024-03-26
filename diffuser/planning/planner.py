import torch
import numpy as np
import math
from scripts.maze_solver import find_path

def plan_waypoints(large_maze, global_start, global_goal, small_maze_size, overlap=None, large_maze_outer_wall=False):
    """Plan a trajectory from start to goal, and return waypoints.
    We work in a coordinate frame where the outer wall is removed.
    The continuous maze is such that the first square is from (0, 0) to (1, 1),
    i.e. the middle of the square is at (0.5, 0.5).
    The planner starts at (1,1)?
    """
    
    # remove the outer wall of the maze
    if large_maze_outer_wall is True:
        global_start = np.array(global_start) - np.array([1, 1])
        global_goal = np.array(global_goal) - np.array([1, 1])
    # print(global_start, global_goal)

    # 
    large_maze = large_maze.copy()
    goal_x = math.ceil(global_goal[0])
    goal_y = math.ceil(global_goal[1])
    large_maze[goal_x][goal_y] = 12
    start_x = math.ceil(global_start[0])
    start_y = math.ceil(global_start[1])
    path = find_path(large_maze, (start_x, start_y), (goal_x, goal_y))

    print("PATH FOUND:")
    print('', path)
    
    # correct maze size
    MAZE_HEIGHT = small_maze_size[0]
    MAZE_WIDTH = small_maze_size[1]
    if overlap is not None:
        MAZE_HEIGHT = int(MAZE_HEIGHT - (overlap[0] * 2))
        MAZE_WIDTH = int(MAZE_WIDTH - (overlap[1] * 2))

    prev_quadrant = find_quadrant(MAZE_HEIGHT, MAZE_WIDTH, global_start)
    prev_quadrant = (abs(prev_quadrant[0]), abs(prev_quadrant[1])) # just in case the start is too close to 0
    prev_square = (-1, -1)
    waypoints = []
    for square in path:
        quadrant = find_quadrant(MAZE_HEIGHT, MAZE_WIDTH, square)
        # if the quadrant changes, add two waypoints
        if quadrant != prev_quadrant:
            # print(f'Quadrant change: {prev_quadrant} -> {quadrant} at {square}')
            prev_quadrant = quadrant
            # waypoints.append([square[0] - 0.5, square[1] - 0.5])
            # waypoints.append([square[0] + 0.5, square[1] + 0.5])
            # -0.5 to get the middle of the square
            # waypoints.append([prev_square[0] - 0.5, prev_square[1] - 0.5])
            # waypoints.append([square[0] - 0.5, square[1] - 0.5])
            noise = 0.05
            waypoints.append([prev_square[0] - np.random.normal(0.5, noise), prev_square[1] - np.random.normal(0.5, noise)])
            waypoints.append([square[0] - np.random.normal(0.5, noise), square[1] - np.random.normal(0.5, noise)])
        prev_square = square

    # add outer wall back to the waypoints
    if large_maze_outer_wall is True:
        waypoints = [[waypoint[0] + 1, waypoint[1] + 1] for waypoint in waypoints]
        # add wall back to the path
        # not necessary since path uses ceil() which adds 1 to the coordinates
        # path = [(square[0] + 1, square[1] + 1) for square in path]
    else:
        path = [(square[0] - 1, square[1] - 1) for square in path]

    return waypoints, path

def find_quadrant(maze_height, maze_width, square):
    # return ((square[0] - 1)//maze_height, (square[1] - 1)//maze_width)
    return ((square[0] - 0.5)//maze_height, (square[1] - 0.5)//maze_width)
    # return square[0] // maze_height, square[1] // maze_width