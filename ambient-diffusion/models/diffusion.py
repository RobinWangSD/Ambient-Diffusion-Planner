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
from modules import MapEncoder

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
        
        
        # TODO: Add agent encoder and denoiser
        # self.agent_encoder = AgentEncoder()
        # self.denoiser = Denoiser()

    def training_step(self, data: Batch, batch_idx: int) -> None:
        # Encode map features using PlanR1's MapEncoder
        map_embeddings = self.map_encoder(data)  # Returns [(M1,...,Mb), hidden_dim]
        
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