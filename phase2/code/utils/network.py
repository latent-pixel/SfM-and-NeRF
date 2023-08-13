import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_features_x=63, num_features_d=27, width=256, depth1=5, depth2=3, depth3=1) -> None:
        super().__init__()
        self.block1 = nn.ModuleList([nn.Linear(num_features_x, width)] + [nn.Linear(width, width) for i in range(depth1 - 1)])
        self.block2 = nn.ModuleList([nn.Linear(num_features_x+width, width)] + [nn.Linear(width, width) for i in range(depth2 - 1)])
        self.output_layer1 = nn.Linear(width, 1)
        self.block3 = nn.ModuleList([nn.Linear(num_features_d+width, width // 2)] + [nn.Linear(width, width) for i in range(depth3 - 1)])
        self.output_layer2 = nn.Linear(width // 2, 3)
    

    def forward(self, x, d):
        x_skip = x
        for layer in self.block1:
            x = F.relu(layer(x))
        
        # concatenating encoded x
        x = torch.cat([x, x_skip], dim=-1)
        for i in range(len(self.block2)):
            layer = self.block2[i]
            if i < len(self.block2)-1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        # obtaining volume density
        density = F.relu(self.output_layer1(x))

        # concatenating encoded d
        x = torch.cat([x, d], dim=-1)
        for layer in self.block3:
            x = F.relu(layer(x))
        
        # obtaining emitted color radiance (rgb) values
        x = F.sigmoid(self.output_layer2(x))

        return torch.cat([x, density], dim=-1)
