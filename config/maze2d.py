import socket
import numpy as np

from diffuser.utils import watch

# ------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ("prefix", ""),
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
]


plan_args_to_watch = [
    ("prefix", ""),
    ##
    ("horizon", "H"),
    ("n_diffusion_steps", "T"),
    ("value_horizon", "V"),
    ("discount", "d"),
    ("normalizer", ""),
    ("batch_size", "b"),
    ##
    ("conditional", "cond"),
]

experiment_args_to_watch = [
    ("experiment_name", ""),
]

# 10k steps per epoch * 200 epochs = 2M steps
base = {
    "diffusion": {
        ## model
        "model": "models.TemporalUnet",
        "diffusion": "models.GaussianDiffusion",
        "horizon": 256,
        "n_diffusion_steps": 256,
        "action_weight": 1,
        "loss_weights": None,
        "loss_discount": 1,
        "predict_epsilon": False,
        "dim_mults": (1, 4, 8),
        "renderer": "utils.Maze2dRenderer",
        ## dataset
        "loader": "datasets.GoalDataset",
        "termination_penalty": None,
        "normalizer": "LimitsNormalizer",
        "preprocess_fns": ["maze2d_set_terminals"],
        "clip_denoised": True,
        "use_padding": False,
        "max_path_length": 40000,
        ## serialization
        "logbase": "logs",
        "prefix": "diffusion/",
        "exp_name": watch(diffusion_args_to_watch),
        ## training
        "n_steps_per_epoch": 10000,
        "loss_type": "l2",
        "n_train_steps": 2e6,
        "batch_size": 128, # 12GB VRAM: 64-256
        "learning_rate": 2e-4,
        "gradient_accumulate_every": 2,
        "ema_decay": 0.995,
        "save_freq": 10000, # in steps
        "sample_freq": 10000,
        "n_saves": 50,
        "save_parallel": False,
        "n_reference": 50,
        "n_samples": 10,
        "bucket": None,
        "device": "cuda",
    },
    "plan": {
        "batch_size": 1,
        "device": "cuda",
        ## diffusion model
        "horizon": 256,
        "n_diffusion_steps": 256,
        "normalizer": "LimitsNormalizer",
        ## serialization
        "vis_freq": 500,
        "logbase": "logs",
        "prefix": "plans/release",
        "exp_name": watch(plan_args_to_watch),
        "suffix": "0",
        "conditional": False,
        ## loading
        "diffusion_loadpath": "f:diffusion/H{horizon}_T{n_diffusion_steps}",
        "diffusion_epoch": "latest",
    },
    # Added for the diffusion planner
    "diffusion_planner": {
        # how many smaller mazes to concatenate to create a larger maze
        "n_maze_h": 2,
        "n_maze_w": 2,
        # if the large maze should have an outer wall
        "large_maze_outer_wall": True,
        # if the small mazes should overlap when combined (i.e. their outer walls are removed)
        "overlap": np.array([1, 1]),
        # If True:  w1 -> w2, w2 -> w3, w3 -> w4, ...
        # If False: w1 -> w2, w3 -> w4, ...
        "overlapping_waypoint_pairs": False,
        # desired trajectory
        "global_start": np.array([1.5, 1.5], dtype=float),
        "global_goal": np.array([19.5, 19.5], dtype=float),
        "waypoints": {
            "global_start": np.array([1.5, 1.5], dtype=float),
            "waypoint1": np.array([7.9, 10.9]), # just at the border if overlap=[1,1]
            "waypoint2": np.array([8.1, 11.1]),
            "global_goal": np.array([19.5, 19.5], dtype=float),
        },
        # plotting
        "plot_conditioning": True, # plot start and end points
        # terminate as soon as the reward is reached # default is False
        "terminate_at_reward": True,
        # if we should terminate if the agent stops moving for 10 steps
        "terminate_if_stuck": False,
        # remove whitespace around image of trajectory
        "_remove_margins": True,
        # remove walls around sub-mazes when plotting large maze
        # image size is 500x500, maze size is 9x12
        # should be [int(500/9), int(500/12)] * overlap
        # "remove_img_margins": None,
        # "remove_img_margins": [int(500/9), int(500/12)], # slight border in between
        "remove_img_margins": [int(500/9)+1, int(500/12)+1],
    }
}

# ------------------------ overrides ------------------------#

"""
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
"""

maze2d_umaze_v1 = {
    "diffusion": {
        "horizon": 128,
        "n_diffusion_steps": 64,
        "batch_size": 256,
    },
    "plan": {
        "horizon": 128,
        "n_diffusion_steps": 64,
    },
}

# maze2d-open-v1 H256_T256
maze2d_open_v0 = {
    **maze2d_umaze_v1,
}

maze2d_large_v1 = {
    "diffusion": {
        "horizon": 384,
        "n_diffusion_steps": 256,
        "batch_size": 64,
    },
    "plan": {
        "horizon": 384,
        "n_diffusion_steps": 256,
    },
}


# ------------------------ diffusion_planner test overrides ------------------------#

# overlap, outer wall
# x: 0[1,8][8,15]16
# y: 0[1,11][11,21]22
maze2d_large_v1_test1 = {
    **maze2d_large_v1,
    "diffusion_planner": {
        # how many smaller mazes to concatenate to create a larger maze
        "n_maze_h": 2,
        "n_maze_w": 2,
        # if the large maze should have an outer wall
        "large_maze_outer_wall": True,
        # if the small mazes should overlap when combined (i.e. their outer walls are removed)
        "overlap": np.array([1, 1]),
        # If True:  w1 -> w2, w2 -> w3, w3 -> w4, ...
        # If False: w1 -> w2, w3 -> w4, ...
        "overlapping_waypoint_pairs": False,
        # desired trajectory
        "global_start": np.array([1.5, 1.5], dtype=float),
        "global_goal": np.array([14.5, 18.5], dtype=float),
    },
}

# no overlap, no outer wall, just concat smaller mazes as they are
# x: [0,9][9,18]
# y: [0,12][12,24]
maze2d_large_v1_test2 = {
    **maze2d_large_v1,
    "diffusion_planner": {
        # if the large maze should have an outer wall
        "large_maze_outer_wall": False,
        # if the small mazes should overlap when combined (i.e. their outer walls are removed)
        "overlap": None,
        "remove_img_margins": None,
        # desired trajectory
        "global_start": np.array([.5, .5], dtype=float),
        "global_goal": np.array([17.5, 23.5], dtype=float),
    },
}

# overlap, but no outer wall
# x: [0,7][7,14]
# y: [0,10][10,20]
maze2d_large_v1_test3 = {
    **maze2d_large_v1,
    "diffusion_planner": {
        # if the large maze should have an outer wall
        "large_maze_outer_wall": False,
        # if the small mazes should overlap when combined (i.e. their outer walls are removed)
        "overlap": np.array([1, 1]),
        # "remove_img_margins": None,
        # "remove_img_margins": [int(500/9), int(500/12)], # slight border in between
        "remove_img_margins": [int(500/9)+1, int(500/12)+1],
        # desired trajectory
        "global_start": np.array([.5, .5], dtype=float),
        "global_goal": np.array([13.5, 19.5], dtype=float),
    },
}

# overlap, no outer wall, 3x3 mazes
maze2d_large_v1_test3 = {
    **maze2d_large_v1,
    "diffusion_planner": {
        # how many smaller mazes to concatenate to create a larger maze
        "n_maze_h": 3,
        "n_maze_w": 3,
        # if the large maze should have an outer wall
        "large_maze_outer_wall": False,
        # if the small mazes should overlap when combined (i.e. their outer walls are removed)
        "overlap": np.array([1, 1]),
        # "remove_img_margins": None,
        # "remove_img_margins": [int(500/9), int(500/12)], # slight border in between
        "remove_img_margins": [int(500/9)+1, int(500/12)+1],
        # desired trajectory
        "global_start": np.array([.5, .5], dtype=float),
        "global_goal": np.array([13.5, 19.5], dtype=float),
    },
}