"""
Inference of diffusion-planner.
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
import sys, os
import copy
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.planning.planner as plan
import diffuser.planning.largemaze2d as maps
import diffuser.planning.diffusing as dm
import diffuser.planning.plotting as plots

from diffuser.planning.logging_utils import suppress_stdout

class Parser(utils.Parser):
    # config file location
    config: str = "config.maze2d"
    # dataset: str = "maze2d-large-v1"
    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    # dataset: str = "maze2d-large-v1-2x2"
    dataset: str = "maze2d-large-v1"
    
class ParserOpen(utils.Parser):
    config: str = "config.maze2d"
    dataset: str = "maze2d-open-v0"

class DiffusionPlanningMazeSolver:
    
    def __init__(self, args, argsdp):
        """Initialize the diffusion planner.
        Load the two diffusion models (main planner and second to fill in the gaps)
        and the open maze environment.
        """
        # ----------------------------- setup and loading -------------------------------#
        print('\n' + ('-' * 25), 'Loading', '-' * 25, flush=True)

        # print('Loading diffusion model at', args.diffusion_loadpath)
        diffusion_experiment = utils.load_diffusion(
            args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
        )

        diffusion = diffusion_experiment.ema
        dataset = diffusion_experiment.dataset
        renderer = diffusion_experiment.renderer
        renderer._remove_margins = argsdp._remove_margins
        # renderer.size_inches = (renderer.size_inches[0] * argsdp.n_maze_h, renderer.size_inches[1]  * argsdp.n_maze_w)

        policy = Policy(diffusion, dataset.normalizer)

        # load the open maze diffusion model to fill in the gaps
        with suppress_stdout():
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

        # write to self
        self.diffusion = diffusion
        self.policy = policy
        self.renderer = renderer
        self.diffusion_open = diffusion_open
        self.policy_open = policy_open
        self.renderer_open = renderer_open
        self.open_maze = open_maze


    def solve_maze(self, args, argsdp):
        """Run diffusion planning on a maze."""
        # set random seed
        np.random.seed(argsdp.seed)
        torch.manual_seed(argsdp.seed)

        # ---------------------------------- generate maze ----------------------------------#
        print('\n' + ('-' * 20), 'Generating the maze', '-' * 20, flush=True)

        # Construct a larger maze by concatenating multiple smaller mazes
        small_maze = datasets.load_environment(args.dataset)
        maze_layout = small_maze.maze_arr
        small_maze_size = maze_layout.shape
        print(f'small maze: {small_maze_size}')
        # print(maze_layout)

        large_maze = maps.generate_large_maze(
            maze_layout=maze_layout, n_maze_h=argsdp.n_maze_h, n_maze_w=argsdp.n_maze_w, overlap=argsdp.overlap, 
            large_maze_outer_wall=argsdp.large_maze_outer_wall
        )
        large_maze_size = large_maze.shape
        argsdp.global_start, argsdp.global_goal = maps.get_start_goal(argsdp.global_start, argsdp.global_goal, large_maze_size)
        print(f'large maze: size {large_maze_size}')
        # print(large_maze)

        # large maze env for better rendering
        large_env = maps.maze_to_gym_env(large_maze)
        large_env.env_name = 'maze2d-custom-v1'

        renderer_large = utils.Maze2dRenderer(large_env)
        renderer_large._remove_margins = argsdp._remove_margins
        renderer_large._bounds = [0, large_maze_size[0], 0, large_maze_size[1]]
        renderer_large.env_name = 'maze2d-custom-v1'
        renderer_large.size_inches = (renderer_large.size_inches[0] * argsdp.n_maze_h, renderer_large.size_inches[1]  * argsdp.n_maze_w)

        # render the maze layout without any trajectory
        # maze_img = plots.render_maze_layout(renderer_large, args.savepath)

        # render the discretized maze layout
        plots.render_discretized_maze_layout(renderer_large, args.savepath)


        # ---------------------------------- main loop planner ----------------------------------#
        print('\n' + ('-' * 20), 'Planning the waypoints', '-' * 20)

        # - Get the planner to output a trajectory in the larger maze
        # - Sample waypoints on the planner trajectory and use them as start and goal locations for the diffuser
        waypoints, path_planner = plan.plan_waypoints(
            large_maze, argsdp.global_start, argsdp.global_goal, small_maze_size, overlap=argsdp.overlap, large_maze_outer_wall=argsdp.large_maze_outer_wall
        )

        # add start and goal
        if not np.allclose(argsdp.global_start, waypoints[0]):
            waypoints = [argsdp.global_start] + waypoints
        if not np.allclose(argsdp.global_goal, waypoints[-1]):
            waypoints = waypoints + [argsdp.global_goal]

        conditions = [np.vstack(copy.deepcopy(waypoints))]

        print(f'Start: {argsdp.global_start} | Goal: {argsdp.global_goal}')
        print(f"Waypoints: {waypoints}")

        # plot the trajectory found by the planner
        planner_traj = np.vstack(path_planner) # should be (timesteps x 2)
        planner_traj = [planner_traj] # might be necessary
        img = renderer_large.composite(
            join(args.savepath, "trajectory_planner.png"), copy.deepcopy(planner_traj), ncol=1, conditions=copy.deepcopy(conditions)
            # join(args.savepath, "trajectory_planner.png"), planner_traj[None], ncol=1, conditions=copy.deepcopy(conditions)
        )
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

        if argsdp.overlapping_waypoint_pairs == True:
            num_steps = len(waypoints) - 1
        else:
            assert len(waypoints) % 2 == 0, f"Number of waypoints should be even, but is {len(waypoints)}"
            num_steps = round(len(waypoints) / 2)

        # TODO optionally add more squares from the planner as conditioning

        traj_pieces = []
        traj_renderings = []
        traj_pieces_gc = []
        for step in range(num_steps):

            print(f"\nDiffusing local traj {step + 1} of {num_steps}")

            for attempt in range(argsdp.max_tries):
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
                if attempt == 0:
                    print(f" start: {local_start_gc} | goal: {local_goal_gc} (global coords)")
                    print(f"        {local_start} |       {local_goal} (local coords)")
                else:
                    print(f" retrying...")

                # check if start and goal are in the same maze
                coord_start = maps.get_maze_coord_from_global_pos(local_start_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                coord_goal = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                assert np.allclose(coord_start, coord_goal), f"start {local_start_gc} and goal {local_goal_gc} should be in the same maze, but are {coord_start} and {coord_goal}"

                # test that the reverse transform works
                # TODO we add 1 too much
                _local_start_gc = maps.local_to_global(local_start, coord_start, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                _local_goal_gc = maps.local_to_global(local_goal, coord_goal, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                assert np.allclose(local_start_gc, _local_start_gc), f"local_start_gc: {local_start_gc} | _local_start_gc: {_local_start_gc}"
                assert np.allclose(local_goal_gc, _local_goal_gc), f"local_goal_gc: {local_goal_gc} | _local_goal_gc: {_local_goal_gc}"

                # generate a trajectory in the local maze
                rollout, rendering = dm.diffuse_trajectory(
                    local_start, local_goal, small_maze, self.diffusion, self.policy, self.renderer, args, argsdp
                )
                distance_from_goal = np.linalg.norm(rollout[-1][:2] - local_goal)  
                print(f' distance from goal: {distance_from_goal:.2f} ({rollout[-1][:2]})')

                if distance_from_goal < argsdp.goal_threshold:
                    break
                else:
                    print(f" Failed to reach goal.")

            # rollout: [time, obs_dim]
            # _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
            _coords = copy.deepcopy(coord_goal)
            traj_pieces.append((np.array(rollout), _coords))
            traj_renderings.append((rendering, _coords))
            # local traj finished

            # convert trajectory to global coordinates
            _t, _c = copy.deepcopy(traj_pieces[-1])
            _t_c = []
            for _p in _t:
                _p_c = maps.local_to_global(_p, _c, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall) 
                _t_c.append(_p_c)
            traj_pieces_gc.append(np.vstack(_t_c))

        # traj = np.vstack([p for p, _ in traj_pieces])
        traj = np.vstack(traj_pieces_gc)
        # print(f"Trajectory: {traj.shape}")

        print(f"\nFinished diffusing the trajectories!")

        # ---------------------------------- plotting ----------------------------------#
        print(f'\nRendering the trajectory.')

        # render the maze layout without any trajectory
        maze_img = plots.render_maze_layout(self.renderer, args.savepath)

        # render the discretized maze layout
        plots.render_discretized_maze_layout(self.renderer, args.savepath)

        plots.render_traj(traj_renderings, args.savepath, empty_img=maze_img, remove_overlap=argsdp.remove_img_margins, add_outer_walls=argsdp.large_maze_outer_wall)


        # -------------------- fill in the gaps with a second diffusion model for open env ----------------------#
        print('\n' + ('-' * 20), 'Diffusing the gaps', '-' * 20)
        # fill in the gaps with maze2d-open-v0 diffuser

        open_maze_size = self.open_maze.maze_arr.shape

        traj_w_filled = []
        # fill in the gap with maze2d-open-v0 diffuser
        for step in range(len(traj_pieces_gc) - 1):
            print(f"\nFilling the gap {step + 1} of {len(traj_pieces_gc) - 1}")

            traj1, traj2 = traj_pieces_gc[step], traj_pieces_gc[step + 1]
            
            # transform to local coordinates
            local_traj1, local_traj2, ltog = maps.global_to_local_openmaze(traj1, traj2, open_maze_size)

            assert np.allclose(local_traj1 + ltog, traj1), f"local_traj1: {local_traj1} | ltog: {ltog} | traj1: {traj1}"
            assert np.allclose(local_traj2 + ltog, traj2), f"local_traj2: {local_traj2} | ltog: {ltog} | traj2: {traj2}"

            # set boundary conditions (optionally add more constraints)
            local_start = local_traj1[-1]
            local_goal = local_traj2[0]

            print(f'start: {traj1[-1]} | goal: {traj2[0]} (global coords)')
            print(f'       {local_start} |       {local_goal} (local open coords)')

            # make sure gap fits within the open maze size
            # assert np.all(np.abs(traj1[-1] - traj2[0]) <  open_maze_size), f"traj1[-1]: {traj1[-1]} | traj2[0]: {traj2[0]} | open_maze_size: {open_maze_size}"
            if not np.all(np.abs(traj1[-1] - traj2[0]) <  open_maze_size): 
                print(f"Gap too large, skipping...")
                print(f"traj1[-1]: {traj1[-1]} | traj2[0]: {traj2[0]} | open_maze_size: {open_maze_size}")
                traj_w_filled.append(traj1)
                # traj_w_filled.append(rollout)
                traj_w_filled.append(traj2)
                continue

            # generate a trajectory in the local maze
            rollout, rendering = dm.diffuse_trajectory(
                local_start, local_goal, self.open_maze, self.diffusion_open, self.policy_open, self.renderer_open, args, argsdp, 
                saveplots=True,
            )
            
            # convert to global coordinates
            rollout = np.asarray(rollout)
            rollout = rollout[:, :2] 
            rollout += ltog

            # rollout: [time, obs_dim]
            # print('shapes', local_traj1.shape, rollout.shape, local_traj2.shape)
            traj_w_filled.append(traj1)
            traj_w_filled.append(rollout)
            traj_w_filled.append(traj2)
            # print(f'traj1[-1] {traj1[-1]} | rollout[0] {rollout[0]}| rollout[-1] {rollout[-1]} | traj2[0] {traj2[0]}')

        print(f"\nFinished diffusing the trajectory gaps!")

        # ---------------------------------- plotting ----------------------------------#

        print(f'\nRendering the trajectory with fillings.')

        traj_wfillings = np.vstack(traj_w_filled)
        print(f"Trajectory with gaps filled in: {traj_wfillings.shape}")
        print(f'  start: {traj_wfillings[0]} | goal: {traj_wfillings[-1]}')
        distance_from_goal = np.linalg.norm(traj_wfillings[-1] - argsdp.global_goal)
        print(f'Distance from goal: {distance_from_goal:.2f}')
        if distance_from_goal > argsdp.goal_threshold:
            print(f"  Failed to reach global_goal.")
        else:
            print(f"  Reached global_goal in {traj_wfillings.shape[0]} steps.")

        # plot final trajectory
        # with waypoints
        img = renderer_large.composite(
            join(args.savepath, "trajectory_wfillings_waypoints.png"), traj_wfillings[None], ncol=1, conditions=copy.deepcopy(conditions)
        )
        # without waypoints
        img = renderer_large.composite(
            join(args.savepath, "trajectory_wfillings.png"), traj_wfillings[None], ncol=1, #conditions=copy.deepcopy(conditions)
        )

        # plots.render_traj(traj_renderings, args.savepath, empty_img=maze_img, remove_overlap=argsdp.remove_img_margins, add_outer_walls=argsdp.large_maze_outer_wall)

        print(f'\nFinished :)')
        return True
    

if __name__ == "__main__":
    with suppress_stdout():
        # Load the arguments 
        args = Parser().parse_args("plan")
        # logger = utils.Logger(args)
        # remove from dataset name so we can load the right pretrained diffusion model
        args.dataset = args.dataset.split("-test")[0]

        # Extra arguments for diffusion planning that weren't in diffuser
        argsdp = Parser().parse_args("diffusion_planner")
        # print(f"argsdp: {argsdp}")


    # argsdp.global_start = np.array([1.5, 2.5])
    # argsdp.global_goal = np.array([10.5, 10.5])

    # argsdp.n_maze_h = 3
    # argsdp.n_maze_w = 3
    # argsdp.global_start = np.array([1.5, 2.5])
    # argsdp.global_goal = np.array([18.5, 20.5], dtype=float)

    # argsdp.n_maze_h = 10
    # argsdp.n_maze_w = 10
    # argsdp.global_start = 'top_right'
    # argsdp.global_goal = 'middle_left'

    # buggy
    # argsdp.n_maze_h = 10
    # argsdp.n_maze_w = 10
    # argsdp.global_start = np.array([1.5, 2.5])
    # argsdp.global_goal = np.array([69.5, 99.5], dtype=float)

    # works top left to bottom right
    # argsdp.n_maze_h = 5
    # argsdp.n_maze_w = 5
    # argsdp.global_start = np.array([1.5, 2.5])
    # argsdp.global_goal = np.array([34.5, 50.5], dtype=float)

    # argsdp.n_maze_h = 5
    # argsdp.n_maze_w = 5
    # argsdp.global_start = np.array([10.5, 45.5])
    # argsdp.global_goal = np.array([30.5, 2.5], dtype=float)

    options = ['top_left', 'top_right', 'middle_left', 'middle_right', 'bottom_left', 'bottom_right']
    # argsdp.n_maze_h = 5
    # argsdp.n_maze_w = 5
    # argsdp.global_start = 'bottom_left'
    # argsdp.global_goal = 'top_right'

    # buggy
    # argsdp.n_maze_h = 3
    # argsdp.n_maze_w = 3
    # argsdp.global_start = np.array([1.5, 2.5])
    # argsdp.global_goal = np.array([10.5, 30.5], dtype=float)

    # np.random.seed(0) # causes a bug
    # argsdp.n_maze_h = 3
    # argsdp.n_maze_w = 3
    # argsdp.global_start = np.array([3.5, 1.5])
    # argsdp.global_goal = np.array([18.5, 30.5], dtype=float)

    # argsdp.add_noise_pos = False
    # argsdp.add_noise_velocity = True

    # argsdp.plan_only = True

    # argsdp.plan_only = False
    # argsdp.replan_every_step = 100

    dplanner = DiffusionPlanningMazeSolver(args, argsdp)

    # run the planner
    argsdp.n_maze_h = 5
    argsdp.n_maze_w = 5
    argsdp.global_start = np.array([1.5, 2.5])
    argsdp.global_goal = np.array([34.5, 50.5], dtype=float)
    argsdp.seed = 0
    dplanner.solve_maze(args, argsdp)