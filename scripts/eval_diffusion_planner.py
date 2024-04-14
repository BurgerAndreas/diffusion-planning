import json
import numpy as np
from os.path import join
import pdb
import imageio
from contextlib import contextmanager
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

from scripts.run_diffusion_planner import DiffusionPlanningMazeSolver, Parser, ParserOpen

"""Run diffusion planner on a set of mazes and measure the success rate."""



def eval():

    with suppress_stdout():
        # Load the arguments 
        args = Parser().parse_args("plan")
        # logger = utils.Logger(args)
        # remove from dataset name so we can load the right pretrained diffusion model
        args.dataset = args.dataset.split("-test")[0]

        # Extra arguments for diffusion planning that weren't in diffuser
        argsdp = Parser().parse_args("diffusion_planner")
        # print(f"argsdp: {argsdp}")

    dplanner = DiffusionPlanningMazeSolver(args, argsdp)

    # Kind of cheating: do not try to execute the plan in the maze env, just assume the plan is correct
    # argsdp.plan_only = False
    # Periodically replan. Slow but might be useful if executing the plan in the maze env deviates from the plan.
    # default: never replan (plan once and execute in the maze env)
    # argsdp.replan_every_step = 100

    successes = []
    for i in range(1):
        with suppress_stdout():
            # Load the arguments 
            args = Parser().parse_args("plan")
            # logger = utils.Logger(args)
            # remove from dataset name so we can load the right pretrained diffusion model
            args.dataset = args.dataset.split("-test")[0]

            # Extra arguments for diffusion planning that weren't in diffuser
            argsdp = Parser().parse_args("diffusion_planner")
            # print(f"argsdp: {argsdp}")

            # TODO: change this to a loop
            # Set the maze size
            argsdp.n_maze_h = 5
            argsdp.n_maze_w = 5
            # Pick a start and end
            options = ['top_left', 'top_right', 'middle_left', 'middle_right', 'bottom_left', 'bottom_right']
            # TODO: check that the start and end do not collide with the maze walls
            argsdp.global_start = 'bottom_left'
            argsdp.global_goal = 'top_right'
            # set a seed
            argsdp.seed = 0

            success = False
            try:
                # Run the planner
                success = dplanner.solve_maze(args, argsdp)
            except Exception as e:
                # print(f"Error: {e}")
                success = False
            successes.append(success)
            print(f"Run {i+1} success: {success}")
    
    print(f"Success rate: {np.mean(successes)}")

if __name__ == "__main__":
    eval()