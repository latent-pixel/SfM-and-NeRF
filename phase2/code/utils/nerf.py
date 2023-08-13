import torch
from .ray_casting import generate_rays, stratified_sampling
from .positional_encoding import positional_encoding
from .network import MLP
from .volume_rendering import volume_rendering


def nerf(height, width, focal_length, extrinsic, near_plane, far_plane,
         init_network, num_samples, Lx, Ld, num_chunks, device):
    # generating rays for each pixel within a frame
    ray_origins, ray_directions = generate_rays(height, width, focal_length, extrinsic)
    # determining query points (what points to sample along a ray)
    query_pts, t_i = stratified_sampling(ray_origins, ray_directions, near_plane, far_plane, num_samples)
    # extending ray directions so that each query point has its corresponding direction 
    query_pt_directions = ray_directions[..., None, :].expand(size=(height, width, num_samples, ray_directions.shape[-1]))

    query_pts = query_pts.reshape((-1, 3))
    query_pt_directions = query_pt_directions.reshape((-1, 3))

    # applying positional encoding
    x_posenc = positional_encoding(query_pts, Lx).to(device)
    d_posenc = positional_encoding(query_pt_directions, Ld).to(device)

    # mlp = MLP(x_posenc.shape[-1], d_posenc.shape[-1], width=128).to(device)

    # chunking the data so that the network doesn't get overwhelmed (although I'm yet to see the advantage)
    x_chunks = torch.chunk(x_posenc, chunks=num_chunks)
    d_chunks = torch.chunk(d_posenc, chunks=num_chunks)

    # predicting color and density from the initialized network, and then reshaping the outputs into images
    rgb_sigma = []
    for i in range(len(x_chunks)):
        rgb_sigma.append(init_network(x_chunks[i], d_chunks[i]))
    rgb_sigma = torch.cat(rgb_sigma, dim=0)
    rgb_sigma = rgb_sigma.reshape((height, width, num_samples, 4))

    # memory_used = torch.cuda.memory_allocated()
    # print("GPU memory footprint: {}".format(round(memory_used / (1024 ** 2), 2)), "MB")
    
    # compositing all the obtained maps, i.e., rendering image from the rays
    rendered_maps = volume_rendering(rgb_sigma, t_i.to(device))
    rgb, depth_map = rendered_maps['rgb'], rendered_maps['depth']
    # print(rgb.shape, depth_map.shape)

    return {'rgb': rgb, 'depth': depth_map}
