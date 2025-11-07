import torch
from torch_geometric.transforms import BaseTransform

from utils import (
    wrap_angle,
)

import matplotlib.pyplot as plt
import numpy as np

Type = {
    0: 'Vehicle', 
    1: 'Pedestrian', 
    2: 'Bicycle'
}

class TargetBuilder(BaseTransform):
    def __init__(
        self, 
        num_historical_steps: int = 20,
        num_future_steps: int = 80, 
        max_agents: int = 20,
        ):
        super(TargetBuilder, self).__init__()
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.max_agents = max_agents

    def __call__(self, data):
        # load data
        data['agent']['position'] = data['agent']['position'][:self.max_agents]    # (A, T=101, (x,y))
        data['agent']['heading'] = data['agent']['heading'][:self.max_agents]      # (A, T)
        data['agent']['velocity'] = data['agent']['velocity'][:self.max_agents]    # (A, T, (vx, vy))
        data['agent']['visible_mask'] = data['agent']['visible_mask'][:self.max_agents]    # (A, T)
        data['agent']['type'] = data['agent']['type'][:self.max_agents]            # (A,)
        data['agent']['id'] = data['agent']['id'][:self.max_agents]
        data['agent']['identity'] = data['agent']['identity'][:self.max_agents]
        data['agent']['box'] = data['agent']['box'][:self.max_agents]
        data['agent']['num_nodes'] = min(self.max_agents, data['agent']['num_nodes'])

        position = data['agent']['position'][:, self.num_historical_steps+1:]   # (A, T_f=80, 2)
        heading = data['agent']['heading'][:, self.num_historical_steps+1:]     # (A, T_f)

        origin = data['agent']['position'][:, self.num_historical_steps]
        theta = data['agent']['heading'][:, self.num_historical_steps]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target_position = torch.bmm(position - origin[:, :2].unsqueeze(1), rot_mat)
        
        target_heading = wrap_angle(heading - theta.unsqueeze(-1))

        data['agent']['target'] = torch.cat(
            [
                target_position,
                torch.cos(target_heading).unsqueeze(-1),
                torch.sin(target_heading).unsqueeze(-1),
            ],
            dim = -1,
        )   # (A, T_f, 4)
        return data
