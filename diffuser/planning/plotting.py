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

def render_traj(global_traj_renderings, savepath, empty_img=None, remove_overlap=None, add_outer_walls=False):
    """stack images based on their coordinates (which small maze they are in)"""

    nrows = max([c[0] for _, c in global_traj_renderings]) + 1
    ncols = max([c[1] for _, c in global_traj_renderings]) + 1

    # print(f"nrows: {nrows} | ncols: {ncols}")
    # print(f'Found coordinates: {[c for _, c in global_traj_renderings]}')

    # img_sample = global_traj_renderings[0][0]
    # wall_color = img_sample[0, 0] # [127 127 127 255]

    # wall_color = [127, 127, 127]

    # 255 white, 0 black, 100 gray
    wall_color = 127

    def add_outer_walls(images):
        # add_outer_walls, same thickness as the overlap
        if add_outer_walls is True:
            # e.g. (776, 832, 4) -> (888, 916, 4)
            # print(f'shape before padding: {images.shape}')
            images = np.pad(images, ((remove_overlap[0], remove_overlap[0]), (remove_overlap[1], remove_overlap[1]), (0,0)), mode='constant', constant_values=wall_color)
            # print(f'shape after padding: {images.shape}')
            images[:remove_overlap[0], :, -1] = 255
            images[-remove_overlap[0]:, :, -1] = 255
            images[:, :remove_overlap[1], -1] = 255
            images[:, -remove_overlap[1]:, -1] = 255
        return images

    if empty_img is None:
        img_sample = global_traj_renderings[0][0]
        empty_img = np.zeros((img_sample.shape), dtype=np.uint8)
    if remove_overlap is not None:
        empty_img = empty_img[remove_overlap[0]:-remove_overlap[0], remove_overlap[1]:-remove_overlap[1]]

    # V1
    images_list = []
    for r in range(nrows):
        for c in range(ncols):
            img_rc = empty_img.copy()
            for img, (row, col) in global_traj_renderings:
                if row == r and col == c:
                    if remove_overlap is not None:
                        img = img[remove_overlap[0]:-remove_overlap[0], remove_overlap[1]:-remove_overlap[1]]
                    img_rc = img
                    break
            images_list.append(img_rc)
    # stack images
    images = np.stack(images_list, axis=0)
    # rearrange
    import einops
    images = einops.rearrange(
        images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrows, ncol=ncols
    )
    images = add_outer_walls(images)
    # save
    img_path = join(savepath, "global_traj_v1.png")
    imageio.imsave(img_path, images)
    print(f'Saving combined trajectory to {img_path}')

    # V2
    # total_image = np.tile(empty_img, (nrows, ncols, 1))
    # for img, (row, col) in global_traj_renderings:
    #     if remove_overlap is not None:
    #         img = img[remove_overlap[0]:-remove_overlap[0], remove_overlap[1]:-remove_overlap[1]]
    #     total_image[
    #         row * img.shape[0] : (row + 1) * img.shape[0],
    #         col * img.shape[1] : (col + 1) * img.shape[1],
    #     ] = img
    # total_image = add_outer_walls(total_image)
    # # save
    # img_path = join(savepath, "global_traj_v2.png")
    # imageio.imsave(img_path, total_image)
    # print(f'Saving combined trajectory to {img_path}')


    # V3 (without any ordering)
    # global_traj_img = np.concatenate(global_traj_renderings, axis=0)
    # global_traj_img = np.concatenate([i for i, _ in global_traj_renderings], axis=0)
    # img_path = join(savepath, "global_traj_v3.png")
    # imageio.imsave(img_path, global_traj_img)
    # print(f'Saving combined trajectory to {img_path}')

    return images

def render_maze_layout(renderer, savepath):
    # render the maze layout without any trajectory
    renderer._plot_obs = False
    dummy = np.array([[0,0,0,0], [0,0,0,0]]) + 1
    maze_img = renderer.composite(
        # array is dummy observation. Need to pass something to get the maze layout
        join(savepath, "maze_layout.png"), [dummy], ncol=1
    )
    renderer._plot_obs = True
    return maze_img

def render_discretized_maze_layout(renderer, savepath):
    # render the discretized maze layout
    renderer._plot_grid = True
    renderer._plot_obs = False
    dummy = np.array([[0,0,0,0], [0,0,0,0]]) + 1
    maze_discrete = renderer.composite(
        # array is dummy observation. Need to pass something to get the maze layout
        join(savepath, "maze_discretized.png"), [dummy], ncol=1
    )
    renderer._plot_obs = True
    renderer._plot_grid = False
    return maze_discrete


if __name__ == "__main__":
    # maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
    
    global_traj_renderings = []
    # red image
    red = np.ones((64, 64, 3), dtype=np.uint8) * 200
    global_traj_renderings.append((red, (0, 0)))
    # blue image
    blue = np.ones((64, 64, 3), dtype=np.uint8) * 100
    global_traj_renderings.append((red, (1, 1)))

    render_traj(global_traj_renderings, "tmp", remove_overlap=0)