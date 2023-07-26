import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_rays(height, width, focal_length, extrinsic):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, -1]
    x, y = torch.meshgrid(torch.arange(width),
                          torch.arange(height),
                          indexing='ij')
    x, y = x.transpose(-1, -2), y.transpose(-1, -2)

    # Normalizing the coordinates 
    transformed_x = (x - width * 0.5) / focal_length
    transformed_y = (y - height * 0.5) / focal_length
    
    # Obtaining ray direction vectors for every pixel in the image
    direction_vecs = torch.stack([transformed_x, -transformed_y, -torch.ones(size=(transformed_x.size()))], dim=-1)
    direction_vecs = direction_vecs[..., None, :]
    world_direction_vecs = direction_vecs * R            # element-wise multiplication of vectors 
    world_direction_vecs = torch.sum(world_direction_vecs, dim=-1) # and adding the corresponding vectors

    # All the rays have the same origin, i.e, the camera position in the world frame
    world_origins = t.expand(world_direction_vecs.size())

    return world_origins, world_direction_vecs