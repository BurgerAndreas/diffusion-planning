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

def render_traj(global_traj_renderings, savepath, remove_overlap=None):
    """stack images based on their coordinates (which small maze they are in)"""

    nrows = max([c[0] for _, c in global_traj_renderings]) + 1
    ncols = max([c[1] for _, c in global_traj_renderings]) + 1

    print(f"nrows: {nrows} | ncols: {ncols}")
    print(f'Found coordinates: {[c for _, c in global_traj_renderings]}')

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
    # save
    img_path = join(savepath, "global_traj_v1.png")
    imageio.imsave(img_path, images)
    print(f'saving v1 to {img_path}')

    # V2
    total_image = np.tile(empty_img, (nrows, ncols, 1))
    for img, (row, col) in global_traj_renderings:
        if remove_overlap is not None:
            img = img[remove_overlap[0]:-remove_overlap[0], remove_overlap[1]:-remove_overlap[1]]
        total_image[
            row * img.shape[0] : (row + 1) * img.shape[0],
            col * img.shape[1] : (col + 1) * img.shape[1],
        ] = img
    # save
    img_path = join(savepath, "global_traj_v2.png")
    imageio.imsave(img_path, total_image)
    print(f'saving v2 to {img_path}')


    # V3 (without any ordering)
    # global_traj_img = np.concatenate(global_traj_renderings, axis=0)
    global_traj_img = np.concatenate([i for i, _ in global_traj_renderings], axis=0)
    img_path = join(savepath, "global_traj_v3.png")
    imageio.imsave(img_path, global_traj_img)
    print(f'saving v3 to {img_path}')

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