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

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

import diffuser.planning.planner as plan
import diffuser.planning.largemaze2d as maps

class Parser(utils.Parser):
    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    config: str = "config.maze2d"
    # dataset: str = "maze2d-large-v1"
    dataset: str = "maze2d-large-v1-test1"

# ---------------------------------- Extra arguments ----------------------------------#

argsdp = Parser().parse_args("diffusion_planner")
# print(f"argsdp: {argsdp}")

# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")

# logger = utils.Logger(args)

# remove from dataset name so we can load the right pretrained diffusion model
args.dataset = args.dataset.split("-test")[0]

# ---------------------------------- loading ----------------------------------#

print('Loading diffusion model at', args.diffusion_loadpath)
diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
renderer._remove_margins = argsdp._remove_margins

policy = Policy(diffusion, dataset.normalizer)

# -------------------- load a second diffusion model for open env ----------------------#

# parser = Parser()
# parser.dataset = "maze2d-open-v0"
# args_open = parser.parse_args("plan")
# args_open.dataset = "maze2d-open-v0"

# diffusion_experiment_open = utils.load_diffusion(
#     args_open.logbase, args_open.dataset, args_open.diffusion_loadpath, epoch=args_open.diffusion_epoch
# )
# diffusion_open = diffusion_experiment_open.ema
# dataset_open = diffusion_experiment_open.dataset

# policy_open = Policy(diffusion_open, dataset_open.normalizer)

# ---------------------------------- main loop planner ----------------------------------#

# TODO(Yifan): Construct a larger maze by concatenating multiple smaller mazes
# might be helful
# https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/maze_model.py
# WALL = 10
# EMPTY = 11
# GOAL = 12
small_maze = datasets.load_environment(args.dataset)
maze_layout = small_maze.maze_arr
small_maze_size = maze_layout.shape
print('maze layout\n', maze_layout)
print('small maze size =', small_maze_size)


large_maze = maps.generate_large_maze(maze_layout=maze_layout, n_maze_h=2, n_maze_w=2, overlap=argsdp.overlap, large_maze_outer_wall=argsdp.large_maze_outer_wall)
print('large maze\n', large_maze)
print('large maze size =', large_maze.shape)
# small_maze_size -= (overlap * 2)

# Maybe better: construct maze as a proper gym / mujoco env that we can render
# via maze_spec https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/pointmaze/maze_model.py


# TODO(Yifan, Jack)
# - Discretize the larger maze into a grid for the planner
# - Get the planner to output a trajectory in the larger maze
# - Sample waypoints on the planner trajectory and use them as start and goal locations for the diffuser
waypoints = plan.plan_waypoints(large_maze, argsdp.global_start, argsdp.global_goal)
if waypoints is None:
    waypoints = argsdp.waypoints
    waypoints['global_start'] = argsdp.global_start
    waypoints['global_goal'] = argsdp.global_goal

print(f"waypoints: {waypoints}")

print(f'\nFinished planning!')

# ---------------------------------- main loop diffusion ----------------------------------#

# TODO(Andreas): how do we deal with planned trajectories that cross maze boundaries?
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
    assert len(waypoints) % 2 == 0
    num_steps = round(len(waypoints) / 2)

# small_maze = datasets.load_environment(args.dataset)
# print(f"Loaded environment {args.dataset}: {small_maze} (type: {type(small_maze)})")

traj_pieces = []
traj_renderings = []
waypoint_list = list(waypoints.keys())
for step in range(num_steps):

    print(f"\nLocal traj {step} of {num_steps}")

    if argsdp.overlapping_waypoint_pairs == True:
        local_start_idx, local_goal_idx = waypoint_list[step], waypoint_list[step + 1]
    else:
        idx = 2 * step
        if idx >= len(waypoint_list):
            break
        local_start_idx, local_goal_idx = waypoint_list[idx], waypoint_list[idx + 1]
    local_start_gc, local_goal_gc = waypoints[local_start_idx], waypoints[local_goal_idx]

    # convert global coordinates to local coordinates
    print(f"start: {local_start_gc} | goal: {local_goal_gc} (global coords)")
    local_start = maps.global_to_local(local_start_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    local_goal = maps.global_to_local(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    print(f"       {local_start} |       {local_goal} (local coords)")

    coord_start = maps.get_maze_coord_from_global_pos(local_start_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    coord_goal = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    assert np.allclose(coord_start, coord_goal), f"start and goal should be in the same maze, but are {coord_start} and {coord_goal}"


    # initial observation includes small non-zero velocity
    # observation = small_maze.reset()

    # if args.conditional:
    #     print("Resetting target to random value")
    #     small_maze.set_target(target_location=None)

    # TODO(Andreas): modify to take in waypoints from planner
    # Set the start and goal locations in the maze env
    # For the diffusion we only need to set the conditioning,
    # but we need the maze for rendering, reward, and terminal checks
    observation = small_maze.reset_to_location(local_start)
    small_maze.set_target(target_location=local_goal)
    
    print(f"Initial observation: {observation}")

    ## set conditioning xy position to be the goal (inpainting)
    target = small_maze._target
    cond = {
        # set velocity to [0, 0]
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }
    print(f"target: {target} | cond: {cond}")

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(small_maze.max_episode_steps):

        state = small_maze.state_vector().copy()
        # print(f"t: {t} | state: {state}")

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            # set the starting point to be the initial obs (inpainting)
            cond[0] = observation

            # plan for the entire horizon, rest of the episode
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
            # print(f"t: {t} | action: {state}")

        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            # if we've reached the end of the sequence, just stay put
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        # action = x_t+1 - x_t + (v_t+1 - v_t)
        # force ~ acceleration = dx + dv
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()

        next_observation, reward, terminal, _ = small_maze.step(action)
        total_reward += reward
        score = small_maze.get_normalized_score(total_reward)
        # print(
        #     f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
        #     f"{action}"
        # )

        if "maze2d" in args.dataset:
            xy = next_observation[:2]
            goal = small_maze.unwrapped._target
            # print(f"maze | pos: {xy} | goal: {goal}")

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if argsdp.terminate_if_stuck and t > 10:
            if np.allclose([np.array(o) for o in rollout[-10:]], atol=1e-2):
                print(f"Stuck at step {t} | R: {total_reward:.2f} | score: {score:.4f}")
                break

        if argsdp.terminate_at_reward and reward > 0:
            terminal = True
            print(f"Reached reward at step {t} | R: {total_reward:.2f} | score: {score:.4f}")

        if (t % args.vis_freq == 0) or terminal or (t == small_maze.max_episode_steps - 1):
            fullpath = join(args.savepath, f"{t}.png")

            if argsdp.plot_conditioning:
                # make conditions into an array with the same length as the number of samples
                conditions = [np.stack(list(cond.values()))]
            else:
                conditions = [None]

            if t == 0:
                renderer.composite(fullpath, samples.observations, ncol=1, conditions=conditions)

            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            img = renderer.composite(
                join(args.savepath, "rollout.png"), np.array(rollout)[None], ncol=1, conditions=conditions
            )
            if terminal or (t == small_maze.max_episode_steps - 1):
                _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                traj_renderings.append((img, _coords))
                print(f'Appended img to global_traj_renderings (size {img.shape}, coords {_coords})')

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    # json_path = join(args.savepath, "rollout.json")
    # json_data = {
    #     "score": score,
    #     "step": t,
    #     "return": total_reward,
    #     "term": terminal,
    #     "epoch_diffusion": diffusion_experiment.epoch,
    # }
    # json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)

    # rollout: [time, obs_dim]
    _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
    traj_pieces.append((np.array(rollout), _coords))
    # local traj finished

print(f"\nFinished diffusing the trajectories!")

# ---------------------------------- fill in the gaps ----------------------------------#
# TODO(Andreas) fill in the gaps with maze2d-open-v0 diffuser

# convert to global coordinates
print(f'\nConverting to global coordinates')
traj_pieces_gc = []
for _t, _c in traj_pieces:
    print('traj local:', len(_t), _t.shape, f'coords: {_c}')
    _t_c = []
    for _p in _t:
        _p_c = maps.local_to_global(_p, _c, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall) 
        _t_c.append(_p_c)
    traj_pieces_gc.append(np.vstack(_t_c))
    print(' global:', len(traj_pieces_gc[-1]), traj_pieces_gc[-1].shape)
print(f"traj_pieces_gc: {len(traj_pieces_gc)} of shape={traj_pieces_gc[0].shape}")

# traj = np.vstack([p for p, _ in traj_pieces])
traj = np.vstack(traj_pieces_gc)
print(f"traj.shape: {traj.shape}")

# # fill in the gap with maze2d-open-v0 diffuser
# open_maze_size = np.array([5, 7])
# for step in range(len(traj_pieces_gc)):
#     traj1, traj2 = traj_pieces_gc[step], traj_pieces_gc[step + 1]
#     # make sure gap fits within the open maze size
#     assert np.all(np.abs(traj1[-1] - traj2[0]) <  open_maze_size), f"traj1[-1]: {traj1[-1]} | traj2[0]: {traj2[0]}"
#     # transform to local coordinates
#     local_traj1, local_traj2 = maps.global_to_local_openmaze(traj1, traj2, open_maze_size)

#     # set conditioning xy position to be the goal (inpainting)
#     local_goal = local_traj2[0]
#     target = local_goal
#     cond = {
#         # set velocity to [0, 0]
#         diffusion_open.horizon - 1: np.array([*target, 0, 0]),
#     }

#     local_start = local_traj1[-1]
#     observation = open_maze.reset_to_location(local_start)
#     open_maze.set_target(target_location=local_goal)

#     # same loop as above
#     # observations for rendering
#     rollout = [observation.copy()]

#     total_reward = 0
#     for t in range(small_maze.max_episode_steps):

#         state = small_maze.state_vector().copy()
#         # print(f"t: {t} | state: {state}")

#         ## can replan if desired, but the open-loop plans are good enough for maze2d
#         ## that we really only need to plan once
#         if t == 0:
#             # set the starting point to be the initial obs (inpainting)
#             cond[0] = observation

#             # plan for the entire horizon, rest of the episode
#             action, samples = policy(cond, batch_size=args.batch_size)
#             actions = samples.actions[0]
#             sequence = samples.observations[0]
#             # print(f"t: {t} | action: {state}")


#         if t < len(sequence) - 1:
#             next_waypoint = sequence[t + 1]
#         else:
#             # if we've reached the end of the sequence, just stay put
#             next_waypoint = sequence[-1].copy()
#             next_waypoint[2:] = 0

#         ## can use actions or define a simple controller based on state predictions
#         # action = x_t+1 - x_t + (v_t+1 - v_t)
#         # force ~ acceleration = dx + dv
#         action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

#         next_observation, reward, terminal, _ = small_maze.step(action)
#         total_reward += reward
#         score = small_maze.get_normalized_score(total_reward)
#         # print(
#         #     f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
#         #     f"{action}"
#         # )

#         if "maze2d" in args.dataset:
#             xy = next_observation[:2]
#             goal = small_maze.unwrapped._target
#             # print(f"maze | pos: {xy} | goal: {goal}")

#         ## update rollout observations
#         rollout.append(next_observation.copy())

#         # logger.log(score=score, step=t)

#         if argsdp.terminate_if_stuck and t > 10:
#             if np.allclose([np.array(o) for o in rollout[-10:]], atol=1e-2):
#                 print(f"Stuck at step {t} | R: {total_reward:.2f} | score: {score:.4f}")
#                 break

#         if argsdp.terminate_at_reward and reward > 0:
#             terminal = True
#             print(f"Reached reward at step {t} | R: {total_reward:.2f} | score: {score:.4f}")

#         if (t % args.vis_freq == 0) or terminal or (t == small_maze.max_episode_steps - 1):
#             fullpath = join(args.savepath, f"{t}.png")

#             if argsdp.plot_conditioning:
#                 # make conditions into an array with the same length as the number of samples
#                 conditions = [np.stack(list(cond.values()))]
#             else:
#                 conditions = [None]

#             if t == 0:
#                 renderer.composite(fullpath, samples.observations, ncol=1, conditions=conditions)

#             ## save rollout thus far
#             img = renderer.composite(
#                 join(args.savepath, "rollout.png"), np.array(rollout)[None], ncol=1, conditions=conditions
#             )
#             if terminal or (t == small_maze.max_episode_steps - 1):
#                 _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
#                 traj_renderings.append((img, _coords))
#                 print(f'Appended img to global_traj_renderings (size {img.shape}, coords {_coords})')

#         if terminal:
#             break

#         observation = next_observation

#     # rollout: [time, obs_dim]
#     # _coords = maps.get_maze_coord_from_global_pos(local_goal_gc, small_maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
#     # traj_pieces.append((np.array(rollout), _coords))
#     # local traj finished


# ---------------------------------- plotting ----------------------------------#
import diffuser.planning.plotting as plots

# render the maze layout without any trajectory
maze_img = plots.render_maze_layout(renderer, args.savepath)

# render the discretized maze layout
plots.render_discretized_maze_layout(renderer, args.savepath)

# TODO(Andreas) add outer walls
plots.render_traj(traj_renderings, args.savepath, empty_img=maze_img, remove_overlap=argsdp.remove_img_margins, add_outer_walls=argsdp.large_maze_outer_wall)
# TODO(Andreas) plot again without trajectory but with constraints (waypoints)
# TODO(Andreas) plot again without trajectory but discretized and with start, end constraint (waypoints)


# large maze env for better rendering
large_env = maps.maze_to_gym_env(large_maze)
