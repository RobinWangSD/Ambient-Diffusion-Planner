import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DistributedSampler
from typing import Dict, List, Tuple, Optional, Iterator
import math
import torch.distributed as dist
from collections import defaultdict

from diffusion_planner.utils.train_utils import opendata


class DiffusionPlannerDataDistributed(Dataset):
    """DataLoader optimized for distributed training with JSON file index"""
    
    def __init__(
        self, 
        data_dir: str, 
        file_index_json: str,
        future_len: int,
        num_historical_steps: int=20, 
        num_future_steps: int=80,
        max_agents: int = 64,
        scenario_types: Optional[List[str]] = None,
        directories: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing data (not used if paths in JSON are absolute)
            file_index_json: Path to the JSON file index created by summary script
            past_neighbor_num: Number of past neighbor agents to include
            predicted_neighbor_num: Number of predicted neighbor agents
            future_len: Length of future trajectory
            scenario_types: Optional list of scenario types to filter (e.g., ['lane_following', 'intersection'])
            directories: Optional list of directories to filter (e.g., ['train_boston', 'train_vegas_1'])
            max_files: Optional maximum number of files to use
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        
        # Load JSON file index
        with open(file_index_json, 'r') as f:
            data = json.load(f)
            self._full_index = data['file_index']
            self.metadata = data['metadata']
        
        # Apply filters
        self._file_index = self._apply_filters(
            self._full_index, 
            scenario_types, 
            directories, 
            max_files
        )
        
        # Create file list and mappings
        self.file_list = list(self._file_index.keys())
        self.idx_to_file = {idx: file for idx, file in enumerate(self.file_list)}
        
        # Organize by directory for I/O optimization
        self._organize_by_directory()
        
        print(f"Dataset initialized with {len(self.file_list):,} files")
        if scenario_types:
            print(f"Filtered by scenario types: {scenario_types}")
        if directories:
            print(f"Filtered by directories: {directories}")
        
    def _apply_filters(
        self, 
        file_index: Dict, 
        scenario_types: Optional[List[str]], 
        directories: Optional[List[str]], 
        max_files: Optional[int]
    ) -> Dict:
        """Apply filters to the file index"""
        filtered = file_index
        
        # Filter by scenario type
        if scenario_types:
            filtered = {
                k: v for k, v in filtered.items() 
                if v['scenario_type'] in scenario_types
            }
        
        # Filter by directory
        if directories:
            filtered = {
                k: v for k, v in filtered.items() 
                if v['dir'] in directories
            }
        
        # Apply max_files limit
        if max_files and len(filtered) > max_files:
            items = list(filtered.items())[:max_files]
            filtered = dict(items)
        
        return filtered
    
    def _organize_by_directory(self):
        """Organize files by their directory for optimized I/O"""
        self.dir_to_indices = defaultdict(list)
        self.idx_to_dir = {}
        
        for idx, file_name in enumerate(self.file_list):
            directory = self._file_index[file_name]['dir']
            
            self.dir_to_indices[directory].append(idx)
            self.idx_to_dir[idx] = directory
        
        self.directories = list(self.dir_to_indices.keys())
        
        # Print statistics
        print(f"Files organized into {len(self.directories)} directories:")
        for dir_name in sorted(self.directories):
            count = len(self.dir_to_indices[dir_name])
            print(f"  {dir_name}: {count:,} files")
    
    def get_scenario_type_distribution(self) -> Dict[str, int]:
        """Get distribution of scenario types in current dataset"""
        distribution = defaultdict(int)
        for file_info in self._file_index.values():
            distribution[file_info['scenario_type']] += 1
        return dict(distribution)
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple:
        file_name = self.idx_to_file[idx]
        file_info = self._file_index[file_name]
        
        # Get file path (use absolute path from index or construct from data_dir)
        file_path = Path(file_info['path'])
        if not file_path.is_absolute() and self.data_dir:
            file_path = self.data_dir / file_path
        
        # Load data
        data = opendata(str(file_path))
        
        # Extract data fields
        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future'][:self._future_len]
        
        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]
        
        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']
        
        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']
        
        static_objects = data['static_objects']
        
        # Include scenario type in output if needed
        output_data = {
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
            "scenario_type": file_info['scenario_type'],  # Add metadata
            "file_name": file_name
        }
        
        return tuple(output_data.values())


class DirectoryAwareDistributedSampler(DistributedSampler):
    """
    Distributed sampler that keeps files from the same directory together
    for optimized I/O while ensuring equal distribution across GPUs
    """
    
    def __init__(
        self,
        dataset: DiffusionPlannerDataDistributed,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        directory_grouping: bool = True
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.directory_grouping = directory_grouping
        
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministic shuffling based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            if self.directory_grouping:
                # Shuffle while keeping directory files together
                indices = self._get_directory_aware_indices(g)
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
    
    def _get_directory_aware_indices(self, generator) -> List[int]:
        """Create indices that keep files from same directory together"""
        dataset = self.dataset
        
        # Shuffle directories
        directories = dataset.directories.copy()
        dir_perm = torch.randperm(len(directories), generator=generator).tolist()
        shuffled_dirs = [directories[i] for i in dir_perm]
        
        # Build indices keeping directories together
        indices = []
        for directory in shuffled_dirs:
            dir_indices = dataset.dir_to_indices[directory].copy()
            # Shuffle within each directory
            dir_perm = torch.randperm(len(dir_indices), generator=generator).tolist()
            shuffled_dir_indices = [dir_indices[i] for i in dir_perm]
            indices.extend(shuffled_dir_indices)
        
        return indices


# Example usage
def create_dataloader(
    file_index_json: str,
    batch_size: int,
    num_workers: int = 4,
    scenario_types: Optional[List[str]] = None,
    directories: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    distributed: bool = False,
    shuffle: bool = True,
    drop_last: bool = True
):
    """
    Create a DataLoader with the JSON file index
    
    Args:
        file_index_json: Path to the JSON file index
        batch_size: Batch size
        num_workers: Number of worker processes
        scenario_types: Optional filter by scenario types
        directories: Optional filter by directories
        max_files: Optional maximum number of files
        distributed: Whether to use distributed training
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    dataset = DiffusionPlannerDataDistributed(
        data_dir=None,  # Paths in JSON are absolute
        file_index_json=file_index_json,
        past_neighbor_num=20,
        predicted_neighbor_num=20,
        future_len=80,
        scenario_types=scenario_types,
        directories=directories,
        max_files=max_files
    )
    
    if distributed:
        sampler = DirectoryAwareDistributedSampler(
            dataset,
            shuffle=shuffle,
            directory_grouping=True
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last
        )
    
    return loader


# Example: Load specific scenario types
if __name__ == "__main__":
    # Create dataloader for specific scenario types
    train_loader = create_dataloader(
        file_index_json="/data/file_index_train.json",
        batch_size=32,
        num_workers=8,
        scenario_types=['lane_following', 'intersection'],  # Filter by scenario type
        directories=['train_boston', 'train_vegas_1'],  # Filter by location
        distributed=True,
        shuffle=True
    )
    
    # Check distribution
    dataset = train_loader.dataset
    print("\nScenario type distribution in filtered dataset:")
    for scenario_type, count in dataset.get_scenario_type_distribution().items():
        print(f"  {scenario_type}: {count:,}")