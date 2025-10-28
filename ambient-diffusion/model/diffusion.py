import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
from copy import deepcopy
import os

class DiffusionPredictor(pl.LightningModule):
    def __init__(self,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
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
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        # # TODO 
        # self.map_encoder = MapEncoder()
        # self.agent_encoder = AgentEncoder()

        # self.denoiser = Denoiser()

    def training_step(self, data: Batch, batch_idx: int) -> None:
        return data

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        pass 

    def on_save_checkpoint(self, checkpoint):
        pass

    def configure_optimizers(self):
        pass