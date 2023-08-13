import torch
import numpy as np


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
    ray_directions = torch.stack([transformed_x, -transformed_y, -torch.ones(size=(transformed_x.size()))], dim=-1)
    ray_directions = ray_directions[..., None, :]
    # Converting to world coordinate frame: element-wise multiplication of vectors and adding the corresponding components
    ray_directions = ray_directions * R             
    ray_directions = torch.sum(ray_directions, dim=-1)

    # All the rays have the same origin, i.e, the camera position in the world frame
    ray_origins = t.expand(ray_directions.size())

    return ray_origins, ray_directions


def stratified_sampling(ray_origins, ray_directions, near_plane, far_plane, num_samples):
    # uniformly sampling at random from equally spaced bins along the ray
    t_i = np.linspace(0.0, 1.0, num_samples+1)
    t_i = near_plane + t_i * (far_plane - near_plane)
    t_i = np.random.uniform(t_i[:-1], t_i[1:])
    t_i = torch.tensor(t_i, dtype=torch.float32)
    
    # point on a ray: r = o + td, we now have t to obtain the points
    t_i = t_i.expand(size = list(ray_directions.shape[:-1]) + [num_samples])
    sampled_points = ray_origins[..., None, :] + t_i[..., None] * ray_directions[..., None, :]
    # sampled_points = sampled_points.view(-1, 3)

    return (sampled_points, t_i)


# stratified_sampling(torch.ones(800, 800, 3), torch.ones(800, 800, 3), 2, 6, 10)
