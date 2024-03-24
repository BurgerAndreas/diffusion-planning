import torch
import numpy as np
import math
from scripts.maze_solver import find_path

def plan_waypoints(large_maze, global_start, global_goal):
    """Plan a trajectory from start to goal, and return waypoints."""
    print(global_start, global_goal)
    large_maze = large_maze.copy()
    goal_x = math.ceil(global_goal[0])
    goal_y = math.ceil(global_goal[1])
    large_maze[goal_x][goal_y] = 12
    path = find_path(large_maze)
    print("PATH FOUND:")
    print(path)

    # "waypoints" = {
    #         "global_start": np.array([1.5, 1.5], dtype=float),
    #         "waypoint1": np.array([7.9, 10.9]), # just at the border if overlap=[1,1]
    #         "waypoint2": np.array([8.1, 11.1]),
    #         "global_goal": np.array([19.5, 19.5], dtype=float),
    #     }
    
    MAZE_HEIGHT = 7
    MAZE_WIDTH = 10
    last_quadrant = (0, 0)
    waypoints = []
    for square in path:
        quadrant = find_quadrant(MAZE_HEIGHT, MAZE_WIDTH, square)
        if quadrant != last_quadrant:
            print(square)
            last_quadrant = quadrant
            waypoints.append([square[0] - 0.5, square[1] - 0.5])
            waypoints.append([square[0] + 0.5, square[1] + 0.5])
    waypoints.pop()
    print("WAYPOINTS:")
    print(waypoints)
    return waypoints

def find_quadrant(maze_height, maze_width, square):
    return ((square[0] - 1)//maze_height, (square[1] - 1)//maze_width)