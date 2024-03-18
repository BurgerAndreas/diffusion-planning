import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    dataset: str = "maze2d-large-v1"
    config: str = "config.maze2d"

"""
Perform inference. 
First you need to train by calling scripts/train_diffuser.py
"""

# ---------------------------------- Extra arguments ----------------------------------#

plot_conditioning = True # plot start and end points

# terminate as soon as the reward is reached # default is False
terminate_at_reward = True

# remove margins from plotted trajectory
_remove_margins = True

local_start, local_goal = None, None
local_start = np.array([1, 1])
local_goal = np.array([8, 11])

# ---------------------------------- setup ----------------------------------#

args = Parser().parse_args("plan")

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)
print(f"Loaded environment {args.dataset}: {env} (type: {type(env)})")

# ---------------------------------- loading ----------------------------------#

print('Loading diffusion model at', args.diffusion_loadpath, end=' ', flush=True)
diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
renderer._remove_margins = _remove_margins

policy = Policy(diffusion, dataset.normalizer)

# ---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:
    print("Resetting target")
    env.set_target()


# set custom start and goal
if local_start is not None:
    observation = env.reset_to_location(local_start)
if local_goal is not None:
    env.set_target(target_location=local_goal)


## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

print(f"Starting at {local_start} | Goal: {local_goal}")

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

        action, samples = policy(cond, batch_size=args.batch_size)
        actions = samples.actions[0]
        sequence = samples.observations[0]
    # pdb.set_trace()

    # 
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

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    # print(
    #     f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
    #     f"a: {action}"
    # )

    if "maze2d" in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(f"       | pos: {xy} | goal: {goal}")

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    if terminate_at_reward and reward > 0:
        terminal = True
        print(f"Reached reward at step {t} | R: {total_reward:.2f} | score: {score:.4f}")

    if (t % args.vis_freq == 0) or terminal or (t == env.max_episode_steps - 1):
        fullpath = join(args.savepath, f"{t}.png")

        if plot_conditioning:
            # make conditions into an array with the same length as the number of samples
            conditions = [np.stack(list(cond.values()))]
        else:
            conditions = [None]

        if t == 0:
            renderer.composite(fullpath, samples.observations, ncol=1, conditions=conditions)

        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(
            join(args.savepath, "rollout.png"), np.array(rollout)[None], ncol=1, conditions=conditions
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
print("Saved numerical results to", json_path)