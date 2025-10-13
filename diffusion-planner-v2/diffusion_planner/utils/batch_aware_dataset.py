import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Dict, List, Tuple, Optional, Iterator
import random
import math
import torch
import torch.distributed as dist

from diffusion_planner.utils.train_utils import opendata


class DiffusionPlannerDataDistributed(Dataset):
    """DataLoader optimized for distributed training with mapping file"""
    
    def __init__(
        self, 
        data_dir: str, 
        mapping_pkl: str,
        past_neighbor_num: int, 
        predicted_neighbor_num: int, 
        future_len: int,
        filter_prefix: Optional[str] = None,
        max_files: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        
        # Load mapping
        with open(mapping_pkl, 'rb') as f:
            self._full_mapping = pickle.load(f)
        
        # Filter mapping if needed
        if filter_prefix:
            self._mapping = {k: v for k, v in self._full_mapping.items() 
                           if k.startswith(filter_prefix)}
        else:
            self._mapping = self._full_mapping
        
        # Apply max_files limit
        if max_files and len(self._mapping) > max_files:
            items = list(self._mapping.items())[:max_files]
            self._mapping = dict(items)
        
        # Create file list
        self.file_list = list(self._mapping.keys())
        self.idx_to_file = {idx: file for idx, file in enumerate(self.file_list)}
        
        # Organize by batch for I/O optimization
        self._organize_by_batch()
        
        print(f"Dataset initialized with {len(self.file_list):,} files")
        
    def _organize_by_batch(self):
        """Organize files by their batch directory"""
        from collections import defaultdict
        
        self.batch_to_indices = defaultdict(list)
        self.idx_to_batch = {}
        
        for idx, file_name in enumerate(self.file_list):
            batch_path = self._mapping[file_name]
            batch_dir = batch_path.split('/')[0]
            
            self.batch_to_indices[batch_dir].append(idx)
            self.idx_to_batch[idx] = batch_dir
        
        self.batch_dirs = list(self.batch_to_indices.keys())
        print(f"Files organized into {len(self.batch_dirs)} batch directories")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple:
        file_name = self.idx_to_file[idx]
        relative_path = self._mapping[file_name]
        file_path = self.data_dir / relative_path
        
        data = opendata(str(file_path))
        
        # Extract data fields
        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']
        
        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]
        
        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']
        
        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']
        
        static_objects = data['static_objects']
        
        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
        }
        
        return tuple(data.values())


class BatchAwareDistributedSampler(DistributedSampler):
    """
    Distributed sampler that tries to keep files from the same batch directory
    together while ensuring equal distribution across GPUs
    """
    
    def __init__(
        self,
        dataset: DiffusionPlannerDataDistributed,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_grouping: bool = True
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_grouping = batch_grouping
        
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministic shuffling based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            if self.batch_grouping:
                # Shuffle while keeping batch directories together
                indices = self._get_batch_aware_indices(g)
            else:
                # Standard shuffling
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def _get_batch_aware_indices(self, generator) -> List[int]:
        """Create indices that keep files from same batch directory together"""
        dataset = self.dataset
        
        # Shuffle batch directories
        batch_dirs = dataset.batch_dirs.copy()
        batch_dir_perm = torch.randperm(len(batch_dirs), generator=generator).tolist()
        shuffled_dirs = [batch_dirs[i] for i in batch_dir_perm]
        
        # Build indices keeping batch directories together
        indices = []
        for batch_dir in shuffled_dirs:
            batch_indices = dataset.batch_to_indices[batch_dir].copy()
            # Shuffle within each batch directory
            batch_perm = torch.randperm(len(batch_indices), generator=generator).tolist()
            shuffled_batch_indices = [batch_indices[i] for i in batch_perm]
            indices.extend(shuffled_batch_indices)
        
        return indices