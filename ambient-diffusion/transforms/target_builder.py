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

        hist_steps = self.num_historical_steps
        fut_steps = self.num_future_steps
        future_start = hist_steps

        position = data['agent']['position']
        heading = data['agent']['heading']
        velocity = data['agent']['velocity']
        visible_mask = data['agent']['visible_mask'].bool()

        data['agent']['history_position'] = position[:, :hist_steps]
        data['agent']['history_heading'] = heading[:, :hist_steps]
        data['agent']['history_velocity'] = velocity[:, :hist_steps]
        data['agent']['history_mask'] = visible_mask[:, :hist_steps]

        current_position = position[:, hist_steps-1]
        current_heading = heading[:, hist_steps-1]
        current_mask = visible_mask[:, hist_steps-1]
        data['agent']['current_states'] = torch.stack(
            [
                current_position[..., 0],
                current_position[..., 1],
                torch.cos(current_heading),
                torch.sin(current_heading),
            ],
            dim=-1,
        )
        data['agent']['current_mask'] = current_mask

        position = position[:, future_start:future_start + fut_steps]   # (A, T_f=80, 2)
        heading = heading[:, future_start:future_start + fut_steps]     # (A, T_f)
        target_valid_mask = visible_mask[:, future_start:future_start + fut_steps]

        # origin = data['agent']['position'][:, hist_steps-1]
        # theta = data['agent']['heading'][:, hist_steps-1]
        # cos, sin = theta.cos(), theta.sin()
        # rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        # rot_mat[:, 0, 0] = cos
        # rot_mat[:, 0, 1] = -sin
        # rot_mat[:, 1, 0] = sin
        # rot_mat[:, 1, 1] = cos
        # target_position = torch.bmm(position - origin[:, :2].unsqueeze(1), rot_mat)
        
        # target_heading = wrap_angle(heading - theta.unsqueeze(-1))

        target = torch.cat(
            [
                position,
                torch.cos(heading).unsqueeze(-1),
                torch.sin(heading).unsqueeze(-1),
            ],
            dim = -1,
        )   # (A, T_f, 4)
        target = target.masked_fill(~target_valid_mask.unsqueeze(-1), 0.0)
        data['agent']['target'] = target
        data['agent']['target_valid_mask'] = target_valid_mask

        # diffusion
        eps = 1e-3 
        t = torch.rand(1, device=position.device) * (1 - eps) + eps
        data['agent']['diffusion_time'] = t.repeat_interleave(data['agent']['num_nodes'])
        return data
