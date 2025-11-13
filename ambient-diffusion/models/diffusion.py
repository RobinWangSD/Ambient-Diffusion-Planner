import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
from copy import deepcopy
import os

from layers import TwoLayerMLP
from modules import (
    MapEncoder,
    AgentEncoder,
    Diffuser,
)

from diffusion_utils import VPSDE_linear

class DiffusionPredictor(pl.LightningModule):
    def __init__(self,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_hops: int = 4,
                 dropout: float = 0.1,
                 agent_time_span: int = 10,
                 agent_radius: float = 60.0,
                 agent_polygon_radius: float = 30.0,
                 agent_num_attn_layers: int = 3,
                 agent_num_heads: int = 8,
                 agent_dropout: float = 0.1,
                 diffuser_num_layers: int = 3,
                 diffuser_num_heads: int = 8,
                 diffuser_dropout: float = 0.1,
                 diffuser_temporal_span: int = 6,
                 diffuser_agent_radius: float = 60.0,
                 diffuser_polygon_radius: float = 30.0,
                 diffuser_segment_length: int = 8,
                 diffuser_segment_overlap: int = 2,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0001,
                 warmup_epochs: int = 4) -> None:
        super(DiffusionPredictor, self).__init__()
        self.save_hyperparameters()

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hops = num_hops
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # Map encoder from PlanR1
        self.map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.agent_encoder = AgentEncoder(
            num_historical_steps=num_historical_steps,
            hidden_dim=hidden_dim,
            time_span=agent_time_span,
            agent_radius=agent_radius,
            polygon_radius=agent_polygon_radius,
            num_attn_layers=agent_num_attn_layers,
            num_heads=agent_num_heads,
            dropout=agent_dropout,
        )
        self.denoiser = Diffuser(
            state_dim=4,
            hidden_dim=hidden_dim,
            num_future_steps=num_future_steps,
            num_historical_steps=num_historical_steps,
            num_layers=diffuser_num_layers,
            num_heads=diffuser_num_heads,
            dropout=diffuser_dropout,
            temporal_span=diffuser_temporal_span,
            agent_radius=diffuser_agent_radius,
            polygon_radius=diffuser_polygon_radius,
            segment_length=diffuser_segment_length,
            segment_overlap=diffuser_segment_overlap,
        )

        # diffusion utils
        self._sde = VPSDE_linear()

    @property
    def sde(self):
        return self._sde

    def training_step(self, data: Batch, batch_idx: int) -> None:
        
        # add noise 
        target = data['agent']['target']        # ([N1, N2, ...], T_f, 4)
        t = data['agent']['diffusion_time']     # ([N1, N2, ...])
        z = torch.randn_like(target, device=target.device)  # ([N1, N2, ...], T_f, 4)
        mean, std = self._sde.marginal_prob(target, t)  
        x_t_noisy = mean + std * z

        agent_store = data['agent']
        num_agents = target.size(0)
        state_dim = target.size(-1)
        start_state = target.new_zeros(num_agents, 1, state_dim)
        if state_dim >= 3:
            start_state[..., 2] = 1.0
        if state_dim >= 4:
            start_state[..., 3] = 0.0

        start_mask = agent_store['current_mask'] if 'current_mask' in agent_store else torch.ones(num_agents, device=target.device, dtype=torch.bool)
        start_state = start_state * start_mask.view(-1, 1, 1).to(start_state.dtype)

        x_t = torch.cat([start_state, x_t_noisy], dim=1)
        z = torch.cat([torch.zeros_like(start_state), z], dim=1)

        # Encode map features using PlanR1's MapEncoder
        map_embeddings = self.map_encoder(data)  # Returns [(M1,...,Mb), hidden_dim]
        agent_embs, visible_mask = self.agent_encoder(data, map_embeddings)

        pred_noise = self.denoiser(
            data=data,
            map_embeddings=map_embeddings,
            agent_embs=agent_embs,
            hist_mask=visible_mask,
            x_t=x_t,
            diffusion_time=t,
        )

        future_mask = self.denoiser.build_future_mask(
            data=data,
            num_agents=target.size(0),
            total_steps=x_t.size(1),
            device=target.device,
        )
        weight = future_mask.unsqueeze(-1).float()
        mse = (pred_noise - z) ** 2
        denom = weight.sum().clamp(min=1.0)
        loss = (mse * weight).sum() / denom

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.size(0) if target.size(0) else 1)

        return loss

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        pass 

    def on_save_checkpoint(self, checkpoint):
        pass

    def configure_optimizers(self):
        pass
