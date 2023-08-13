import torch


def volume_rendering(mlp_output, t_i):
    # distance between adjacent samples
    delta = t_i[..., 1:] - t_i[..., :-1]    # H x W x (NumSamples-1)
    delta = torch.cat([delta, 1e10 * torch.ones_like(delta[..., 0:1])], dim=-1) # H x W x NumSamples

    # calculating alpha (transparency)
    alpha = 1. - torch.exp(-mlp_output[..., -1] * delta) # H x W x NumSamples

    # accumulated transmittance
    T_temp = 1 - alpha + 1e-10
    T = torch.cumprod(T_temp, dim=-1) 
    T = torch.roll(T, 1, -1)
    T[..., 0] = 1.  # H x W x NumSamples

    # calculating rgb values
    weights = T * alpha # H x W x NumSamples
    rgb = torch.sum(weights[..., None] * mlp_output[..., :-1], dim=-2) # H x W x 3
    acc_alpha = torch.sum(weights, dim=-1) # accumulated alpha map (for compositing on to a white background)
    rgb = rgb + (1 - acc_alpha[..., None])

    # depth map is nothing but weights at each query point
    depth_map = torch.sum(weights * t_i, dim=-1)

    return {'rgb': rgb, 'depth': depth_map}
