import torch


def positional_encoding(x, L):
    # x = x.view(-1, 3)
    freq = torch.tensor([2**i for i in range(L)]) 
    pos_sin = torch.sin(freq[..., None] * x.unsqueeze(dim=1))
    pos_cos = torch.cos(freq[..., None] * x.unsqueeze(dim=1))

    pos = torch.cat([pos_sin, pos_cos], dim=-1)
    pos = pos.view(x.shape[0], -1)
    pos = torch.cat([x, pos], dim=-1)

    return pos
