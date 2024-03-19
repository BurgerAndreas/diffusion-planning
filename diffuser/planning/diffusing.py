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

"""
based on main loop from diffuser/scripts/run_maze2d.py
"""


def diffuse_trajectory(start, goal, maze, diffusion, policy, renderer, args, argsdp):
    # initial observation includes small non-zero velocity
    # observation = maze.reset()

    # if args.conditional:
    #     print("Resetting target to random value")
    #     maze.set_target(target_location=None)

    # Modify to take in waypoints from planner
    # Set the start and goal locations in the maze env
    # For the diffusion we only need to set the conditioning,
    # but we need the maze for rendering, reward, and terminal checks
    observation = maze.reset_to_location(start)
    maze.set_target(target_location=goal)
    
    print(f"Initial observation: {observation}")

    ## set conditioning xy position to be the goal (inpainting)
    target = maze._target
    cond = {
        # set velocity to [0, 0]
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }
    print(f"target: {target} | cond: {cond}")

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(maze.max_episode_steps):

        state = maze.state_vector().copy()
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

        next_observation, reward, terminal, _ = maze.step(action)
        total_reward += reward
        score = maze.get_normalized_score(total_reward)
        # print(
        #     f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
        #     f"{action}"
        # )

        if "maze2d" in args.dataset:
            xy = next_observation[:2]
            goal = maze.unwrapped._target
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

        if (t % args.vis_freq == 0) or terminal or (t == maze.max_episode_steps - 1):
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
            if terminal or (t == maze.max_episode_steps - 1):
                # _coords = maps.get_maze_coord_from_global_pos(goal_gc, maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
                # traj_renderings.append((img, _coords))
                # print(f'Appended img to global_traj_renderings (size {img.shape}, coords {_coords})')
                traj_rendering = img

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
    return rollout, traj_rendering

# Usage
# _coords = maps.get_maze_coord_from_global_pos(goal_gc, maze_size, argsdp.overlap, argsdp.large_maze_outer_wall)
# traj_pieces.append((np.array(rollout), _coords))