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
        self.agent_encoder = AgentEncoder()
        self.denoiser = Diffuser()

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
        x_t = mean + std * z

        # Encode map features using PlanR1's MapEncoder
        map_embeddings = self.map_encoder(data)  # Returns [(M1,...,Mb), hidden_dim]
        agent_embs, visible_mask = self.agent_encoder(data, map_embeddings)
        # TODO: Implement diffusion training step
        # - Use map_embeddings in your denoising process
        # - Implement forward diffusion (add noise)
        # - Implement reverse diffusion (denoise)
        # - Calculate loss
        
        loss = None  # Replace with actual loss computation
        return loss

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        pass 

    def on_save_checkpoint(self, checkpoint):
        pass

    def configure_optimizers(self):
        pass
