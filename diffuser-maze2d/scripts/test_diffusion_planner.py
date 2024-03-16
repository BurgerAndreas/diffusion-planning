"""
Inference / validation of diffusion-planner!
First you need to train the diffuser by calling diffuser-maze2d/scripts/train.py

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

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    # maze2d_umaze_v1 maze2d_medium_v1 maze2d_large_v1
    dataset: str = "maze2d-umaze-v1"
    config: str = "config.maze2d"

# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")

# logger = utils.Logger(args)

# TODO(Yifan): Construct a larger maze by concatenating multiple smaller mazes
large_maze = None

# ---------------------------------- loading ----------------------------------#

print('Loading diffusion model at', args.diffusion_loadpath, end=' ', flush=True)
diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

# ---------------------------------- main loop planner ----------------------------------#

# TODO(Yifan, Jack)
# - Discretize the larger maze into a grid for the planner
# - Get the planner to output a trajectory in the larger maze
# - Sample waypoints on the planner trajectory and use them as start and goal locations for the diffuser
waypoints = {
    "global_start": np.array([0, 0]),
    "waypoint1": np.array([5, 5]),
    "waypoint2": np.array([7, 7]),
    "global_goal": np.array([10, 10]),
}

# ---------------------------------- main loop diffusion ----------------------------------#

# TODO(Andreas): how do we deal with waypoints that cross maze boundaries?

for local_start, local_goal in zip(waypoints.values(), waypoints.values()):
    print(f"local_start: {local_start} | local_goal: {local_goal}")

    small_maze = datasets.load_environment(args.dataset)
    print(f"Loaded environment {args.dataset}: {small_maze} (type: {type(small_maze)})")

    observation = small_maze.reset()

    # TODO(Andreas): modify to take in waypoints from planner
    if args.conditional:
        print("Resetting target")
        small_maze.set_target()

    ## set conditioning xy position to be the goal
    target = small_maze._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(small_maze.max_episode_steps):

        state = small_maze.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation

            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t + 1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
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
        print(
            f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
            f"{action}"
        )

        if "maze2d" in args.dataset:
            xy = next_observation[:2]
            goal = small_maze.unwrapped._target
            print(f"maze | pos: {xy} | goal: {goal}")

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f"{t}.png")

            if t == 0:
                renderer.composite(fullpath, samples.observations, ncol=1)

            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(
                join(args.savepath, "rollout.png"), np.array(rollout)[None], ncol=1
            )

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(args.savepath, "rollout.json")
    json_data = {
        "score": score,
        "step": t,
        "return": total_reward,
        "term": terminal,
        "epoch_diffusion": diffusion_experiment.epoch,
    }
    json.dump(json_data, open(json_path, "w"), indent=2, sort_keys=True)
