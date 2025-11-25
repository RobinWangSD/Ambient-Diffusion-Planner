from typing import Optional
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import os

from datasets import NuplanDataset
from transforms import TargetBuilder

class NuplanDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 train_metadata_path: str,
                 val_metadata_path: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 max_agents: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 
                 **kwargs) -> None:
        super(NuplanDataModule, self).__init__()
        self.root = root
        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        # TODO: add data augmentation 
        self.train_transform = TargetBuilder(num_historical_steps, num_future_steps, max_agents)
        self.val_transform = TargetBuilder(num_historical_steps, num_future_steps, max_agents)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NuplanDataset(self.root, self.train_metadata_path, self.train_transform, data_type='train')
        self.val_dataset = NuplanDataset(self.root, self.val_metadata_path, self.val_transform, data_type='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

