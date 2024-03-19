"""
Inference / validation of diffusion-planner!
First you need to train the diffuser by calling scripts/train_diffuser.py

Idea:
- Load the diffusion model (e.g. trained on maze2d-large-v1) as is, without any additional training
- Construct a larger maze by concatenating multiple smaller mazes
- Discretize the larger maze into a grid for the planner
- Get the planner to output a trajectory in the larger maze
- Sample waypoints on the planner trajectory and use them as start and goal locations for the diffuser
- Diffuser only acts on mazes the size it was trained on 
"""

import json
import numpy as np
from os.path import join
import pdb
import imageio
from contextlib import contextmanager
import sys, os
import copy

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.planning.planner as plan
import diffuser.planning.largemaze2d as maps
import diffuser.planning.diffusing as dm


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# ----------------------------- setup and loading -------------------------------#
class Parser(utils.Parser):
    # config file location
    config: str = "config.maze2d"
    # dataset: str = "maze2d-large-v1"
    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    dataset: str = "maze2d-large-v1-test1"


with suppress_stdout():
    # Load the arguments 
    args = Parser().parse_args("plan")
    # logger = utils.Logger(args)
    # remove from dataset name so we can load the right pretrained diffusion model
    args.dataset = args.dataset.split("-test")[0]

    # Extra arguments for diffusion planning that weren't in diffuser
    argsdp = Parser().parse_args("diffusion_planner")
    # print(f"argsdp: {argsdp}")

    # print('Loading diffusion model at', args.diffusion_loadpath)
    diffusion_experiment = utils.load_diffusion(
        args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
    )

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer
    renderer._remove_margins = argsdp._remove_margins

    policy = Policy(diffusion, dataset.normalizer)


# ---------------------------------- generate maze ----------------------------------#
    
# Construct a larger maze by concatenating multiple smaller mazes
small_maze = datasets.load_environment(args.dataset)
maze_layout = small_maze.maze_arr
small_maze_size = maze_layout.shape
print(f'small maze (size {small_maze_size})\n', maze_layout)

large_maze = maps.generate_large_maze(
    maze_layout=maze_layout, n_maze_h=argsdp.n_maze_h, n_maze_w=argsdp.n_maze_w, overlap=argsdp.overlap, 
    large_maze_outer_wall=argsdp.large_maze_outer_wall
)
print(f'large maze (size {large_maze.shape})\n', large_maze)

# ---------------------------------- main loop planner ----------------------------------#
print('\n' + ('-' * 20), 'Planning the waypoints', '-' * 20)

# TODO(Yifan, Jack)
# - Get the planner to output a trajectory in the larger maze
# - Sample waypoints on the planner trajectory and use them as start and goal locations for the diffuser
waypoints = plan.plan_waypoints(large_maze, argsdp.global_start, argsdp.global_goal)
if waypoints is None:
    waypoints = argsdp.waypoints
    waypoints['global_start'] = argsdp.global_start
    waypoints['global_goal'] = argsdp.global_goal

waypoints = [v for k, v in waypoints.items()]
print(f"waypoints: {waypoints}")

print(f'\nFinished planning!')

# ---------------------------------- main loop diffusion ----------------------------------#
print('\n' + ('-' * 20), 'Diffusing the trajectories', '-' * 20)

# How do we deal with planned trajectories that cross maze boundaries?
# Idea 1: sample two waypoints right next to each other at the boundary, one in each maze
# diffuser does not 'see' the walls. But all data the diffuser is trained from avoids the walls.
# i.e. just setting a waypoint on the boundary will be out of distribution and fail.
# e.g. this is the closest to the wall the diffuser will every go: 
# maze | pos: [ 0.99922423 10.19801523] | goal: [ 1 11]
# Idea 2: Instead we could let the mazes overlap a bit, 
# so that a waypoint just inside the outer walls are actually at the boundary to the next maze

conditions = [np.vstack(copy.deepcopy(waypoints))]
if argsdp.overlapping_waypoint_pairs == True:
    num_steps = len(waypoints) - 1
else:
    assert len(waypoints) % 2 == 0
    num_steps = round(len(waypoints) / 2)


traj_pieces = []
traj_renderings = []
for step in range(num_steps):

    print(f"\nDiffusing local traj {step + 1} of {num_steps}")

    if argsdp.overlapping_waypoint_pairs == True:
        local_start_idx, local_goal_idx = step, step + 1
    else:
        idx = 2 * step
        if idx >= len(waypoints):
            break
        local_start_idx, local_goal_idx = idx, idx + 1
    local_start_gc = copy.deepcopy(waypoints[local_start_idx])
    local_goal_gc = copy.deepcopy(waypoints[local_goal_idx])

    # convert global coordinates to local coordinates
    local_start = maps.global_to_local(local_start_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    local_goal = maps.global_to_local(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    print(f" start: {local_start_gc} | goal: {local_goal_gc} (global coords)")
    print(f"        {local_start} |       {local_goal} (local coords)")

    # check if start and goal are in the same maze
    coord_start = maps.get_maze_coord_from_global_pos(local_start_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    coord_goal = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    assert np.allclose(coord_start, coord_goal), f"start and goal should be in the same maze, but are {coord_start} and {coord_goal}"

    # generate a trajectory in the local maze
    rollout, rendering = dm.diffuse_trajectory(local_start, local_goal, small_maze, diffusion, policy, renderer, args, argsdp)

    # rollout: [time, obs_dim]
    _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    traj_pieces.append((np.array(rollout), _coords))
    traj_renderings.append((rendering, _coords))
    # local traj finished

# convert trajectory so far to global coordinates
print(f"\nFinished diffusing the trajectories!")
print(f'Converting trajectory pieces to global coordinates.')
traj_pieces_gc = []
for _t, _c in traj_pieces:
    _t_c = []
    for _p in _t:
        _p_c = maps.local_to_global(_p, _c, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall) 
        _t_c.append(_p_c)
    traj_pieces_gc.append(np.vstack(_t_c))

# traj = np.vstack([p for p, _ in traj_pieces])
traj = np.vstack(traj_pieces_gc)
# print(f"Trajectory: {traj.shape}")


# ---------------------------------- plotting ----------------------------------#
import diffuser.planning.plotting as plots

print(f'\nRendering the trajectory.')

# render the maze layout without any trajectory
maze_img = plots.render_maze_layout(renderer, args.savepath)

# render the discretized maze layout
plots.render_discretized_maze_layout(renderer, args.savepath)

plots.render_traj(traj_renderings, args.savepath, empty_img=maze_img, remove_overlap=argsdp.remove_img_margins, add_outer_walls=argsdp.large_maze_outer_wall)
# TODO(Andreas) plot again without trajectory but with constraints (waypoints)
# TODO(Andreas) plot again without trajectory but discretized and with start, end constraint (waypoints)



# -------------------- fill in the gaps with a second diffusion model for open env ----------------------#
print('\n' + ('-' * 20), 'Diffusing the gaps', '-' * 20)
# fill in the gaps with maze2d-open-v0 diffuser

with suppress_stdout():
    class ParserOpen(utils.Parser):
        config: str = "config.maze2d"
        dataset: str = "maze2d-open-v0"

    args_open = ParserOpen().parse_args("plan")
    args_open.dataset = "maze2d-open-v0"

    diffusion_experiment_open = utils.load_diffusion(
        args_open.logbase, args_open.dataset, args_open.diffusion_loadpath, epoch=args_open.diffusion_epoch
    )
    diffusion_open = diffusion_experiment_open.ema
    dataset_open = diffusion_experiment_open.dataset
    renderer_open = diffusion_experiment_open.renderer
    renderer_open._remove_margins = argsdp._remove_margins

    policy_open = Policy(diffusion_open, dataset_open.normalizer)

    open_maze = datasets.load_environment(args_open.dataset)
    open_maze_layout = open_maze.maze_arr
    open_maze_size = open_maze_layout.shape

traj_w_filled = []
# fill in the gap with maze2d-open-v0 diffuser
for step in range(len(traj_pieces_gc) - 1):
    print(f"\nFilling the gap {step + 1} of {len(traj_pieces_gc) - 1}")

    traj1, traj2 = traj_pieces_gc[step], traj_pieces_gc[step + 1]
    # make sure gap fits within the open maze size
    assert np.all(np.abs(traj1[-1] - traj2[0]) <  open_maze_size), f"traj1[-1]: {traj1[-1]} | traj2[0]: {traj2[0]}"
    
    # transform to local coordinates
    local_traj1, local_traj2, ltog = maps.global_to_local_openmaze(traj1, traj2, open_maze_size)

    # set boundary conditions (optionally add more constraints)
    local_start = local_traj1[-1]
    local_goal = local_traj2[0]

    # print(f'start: {traj1[-1]} | goal: {traj2[0]} (global coords)')
    # print(f'       {local_start} |       {local_goal} (local open coords)')

    # generate a trajectory in the local maze
    rollout, rendering = dm.diffuse_trajectory(
        local_start, local_goal, open_maze, diffusion_open, policy_open, renderer_open, args, argsdp, saveplots=False
    )
    
    # # convert to global coordinates
    rollout = np.asarray(rollout)
    rollout = rollout[:, :2] 
    rollout += ltog

    # rollout: [time, obs_dim]
    # print('shapes', local_traj1.shape, rollout.shape, local_traj2.shape)
    traj_w_filled.append(traj1)
    traj_w_filled.append(rollout)
    traj_w_filled.append(traj2)

print(f"\nFinished diffusing the trajectory gaps!")

# ---------------------------------- plotting ----------------------------------#
import diffuser.planning.plotting as plots

print(f'\nRendering the trajectory with fillings.')

traj_wfillings = np.vstack(traj_w_filled)
print(f"Trajectory with gaps filled in: {traj_wfillings.shape}")
print(f'  start: {traj_wfillings[0]} | goal: {traj_wfillings[-1]}')

# large maze env for better rendering
large_maze_size = large_maze.shape
large_env = maps.maze_to_gym_env(large_maze)
large_env.env_name = 'maze2d-custom-v1'

renderer_large = utils.Maze2dRenderer(large_env)
renderer_large._bounds = [0, large_maze_size[0], 0, large_maze_size[1]]
renderer_large.env_name = 'maze2d-custom-v1'

# render the maze layout without any trajectory
# maze_img = plots.render_maze_layout(renderer_large, args.savepath)

# render the discretized maze layout
plots.render_discretized_maze_layout(renderer_large, args.savepath)

img = renderer_large.composite(
    join(args.savepath, "trajectory_wfillings.png"), traj_wfillings[None], ncol=1, #conditions=conditions
)

# print(f"conditions: {conditions}")
img = renderer_large.composite(
    join(args.savepath, "trajectory_wfillings_waypoints.png"), traj_wfillings[None], ncol=1, conditions=conditions
)

# plots.render_traj(traj_renderings, args.savepath, empty_img=maze_img, remove_overlap=argsdp.remove_img_margins, add_outer_walls=argsdp.large_maze_outer_wall)



print(f'\nFinished :)')