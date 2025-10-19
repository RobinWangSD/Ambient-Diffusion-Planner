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
        max_agents: int = 60,
        max_lanes: int = 80,
        max_crosswalks: int = 5,
        max_drivable_area_segments: int = 30,
        max_static_objects: int = 30,
        radius: float = 120,
        margin: float = 50,
        scenario_types: Optional[List[str]] = None,
        directories: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing data (not used if paths in JSON are absolute)
            file_index_json: Path to the JSON file index created by summary script
            future_len: Length of future trajectory
            num_historical_steps: Number of historical timesteps (default 20)
            num_future_steps: Number of future timesteps (default 80)
            max_agents: Maximum number of agents to include
            max_lanes: Maximum number of lanes 
            max_crosswalks: Maximum number of crosswalks
            max_drivable_area_segments: Maximum number of drivable area segments
            max_static_objects: Maximum number of static objects
            radius: Radius for map queries
            margin: Margin for filtering map objects
            scenario_types: Optional list of scenario types to filter
            directories: Optional list of directories to filter
            max_files: Optional maximum number of files to use
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._future_len = future_len
        self._num_historical_steps = num_historical_steps
        self._num_future_steps = num_future_steps
        self._max_agents = max_agents
        self._max_lanes = max_lanes
        self._max_crosswalks = max_crosswalks
        self._max_drivable_area_segments = max_drivable_area_segments
        self._max_static_objects = max_static_objects
        self._radius = radius
        self._margin = margin
        
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

    def _convert_new_to_old_format(self, data: Dict) -> Dict:
        """
        Convert new format (from get_features) to old format expected by model
        
        Old format expects:
        - ego_current_state: (10,) - x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate
        - ego_agent_future: (future_len, 3) - relative future positions
        - neighbor_agents_past: (num_agents, 21, 11) - past agent states
        - neighbor_agents_future: (num_agents, future_len, 3) - future agent states
        - static_objects: (num_static, 10) - static object features
        - lanes: (lane_num, lane_len, 12) - lane polylines with features
        - route_lanes: (route_num, route_len, 12) - route lane polylines
        - lanes_speed_limit: (lane_num, 1) - speed limits
        - lanes_has_speed_limit: (lane_num, 1) - speed limit flags
        - route_lanes_speed_limit: (route_num, 1)
        - route_lanes_has_speed_limit: (route_num, 1)
        """
        
        old_format = {}
        
        # Extract agent data
        agent_data = data.get('agent', {})
        num_agents = agent_data.get('num_nodes', 1)
        
        # 1. Ego current state - using present (timestep 20) ego data
        # Format: [x, y, cos(heading), sin(heading), vx, vy, ax, ay, steering_angle, yaw_rate]
        ego_idx = 0  # Ego is first agent
        if num_agents > 0:
            # Get ego at present time (index 20 in historical)
            present_idx = self._num_historical_steps
            ego_pos = agent_data['position'][ego_idx, present_idx] if 'position' in agent_data else np.zeros(2)
            ego_heading = agent_data['heading'][ego_idx, present_idx] if 'heading' in agent_data else 0.0
            ego_vel = agent_data['velocity'][ego_idx, present_idx] if 'velocity' in agent_data else np.zeros(2)
            
            ego_current_state = np.zeros(10, dtype=np.float32)
            ego_current_state[0] = 0.0  # x (ego-centric, so 0)
            ego_current_state[1] = 0.0  # y (ego-centric, so 0) 
            ego_current_state[2] = np.cos(ego_heading)
            ego_current_state[3] = np.sin(ego_heading)
            ego_current_state[4] = ego_vel[0]  # vx
            ego_current_state[5] = ego_vel[1]  # vy
            # ax, ay, steering_angle, yaw_rate would need to be computed from trajectory
            old_format['ego_current_state'] = ego_current_state
        else:
            old_format['ego_current_state'] = np.zeros(10, dtype=np.float32)
        
        # 2. Ego future trajectory (relative positions)
        if num_agents > 0 and 'position' in agent_data:
            future_start = self._num_historical_steps + 1
            future_end = min(future_start + self._future_len, agent_data['position'].shape[1])
            ego_future = agent_data['position'][ego_idx, future_start:future_end]
            
            # Pad if needed
            if len(ego_future) < self._future_len:
                padding = np.repeat(ego_future[-1:], self._future_len - len(ego_future), axis=0)
                ego_future = np.concatenate([ego_future, padding], axis=0)
            
            old_format['ego_agent_future'] = ego_future[:self._future_len].astype(np.float32)
        else:
            old_format['ego_agent_future'] = np.zeros((self._future_len, 2), dtype=np.float32)
        
        # 3. Neighbor agents past (last 21 timesteps including present)
        neighbor_agents_past = np.zeros((self._num_agents, 21, 11), dtype=np.float32)
        
        if num_agents > 1:  # Has neighbors
            for i in range(1, min(num_agents, self._num_agents + 1)):
                for t in range(21):
                    hist_idx = t  # Use first 21 timesteps
                    if hist_idx < agent_data['position'].shape[1]:
                        # Position (x, y)
                        neighbor_agents_past[i-1, t, 0:2] = agent_data['position'][i, hist_idx]
                        # Heading (cos, sin)
                        heading = agent_data['heading'][i, hist_idx]
                        neighbor_agents_past[i-1, t, 2] = np.cos(heading)
                        neighbor_agents_past[i-1, t, 3] = np.sin(heading)
                        # Velocity
                        neighbor_agents_past[i-1, t, 4:6] = agent_data['velocity'][i, hist_idx]
                        # Box dimensions (width, length)
                        neighbor_agents_past[i-1, t, 6] = agent_data['box'][i, 2] * 2  # width
                        neighbor_agents_past[i-1, t, 7] = agent_data['box'][i, 0] + agent_data['box'][i, 1]  # length
                        # Agent type encoding (one-hot: vehicle, pedestrian, bicycle)
                        agent_type = agent_data['type'][i]
                        if agent_type == 0:  # vehicle
                            neighbor_agents_past[i-1, t, 8:11] = [1, 0, 0]
                        elif agent_type == 1:  # pedestrian
                            neighbor_agents_past[i-1, t, 8:11] = [0, 1, 0]
                        else:  # bicycle
                            neighbor_agents_past[i-1, t, 8:11] = [0, 0, 1]
        
        old_format['neighbor_agents_past'] = neighbor_agents_past
        
        # 4. Neighbor agents future
        neighbor_agents_future = np.zeros((self._num_agents, self._future_len, 3), dtype=np.float32)
        
        if num_agents > 1:
            for i in range(1, min(num_agents, self._num_agents + 1)):
                future_start = self._num_historical_steps + 1
                future_end = min(future_start + self._future_len, agent_data['position'].shape[1])
                
                for t in range(future_end - future_start):
                    pos = agent_data['position'][i, future_start + t]
                    heading = agent_data['heading'][i, future_start + t]
                    neighbor_agents_future[i-1, t, 0:2] = pos
                    neighbor_agents_future[i-1, t, 2] = heading
        
        old_format['neighbor_agents_future'] = neighbor_agents_future
        
        # 5. Static objects
        static_objects = np.zeros((self._static_objects_num, 10), dtype=np.float32)
        
        # Extract static objects from polygon data
        polygon_data = data.get('polygon', {})
        if polygon_data and 'type' in polygon_data:
            static_mask = polygon_data['type'] == 3  # STATIC_OBJECT type
            static_indices = np.where(static_mask)[0][:self._static_objects_num]
            
            for i, idx in enumerate(static_indices):
                # Position
                static_objects[i, 0:2] = polygon_data['position'][idx]
                # Heading (cos, sin)
                if polygon_data.get('heading_valid_mask', [False])[idx]:
                    heading = polygon_data['heading'][idx]
                    static_objects[i, 2] = np.cos(heading)
                    static_objects[i, 3] = np.sin(heading)
                # Box dimensions - would need to be extracted from polyline data
                # Type encoding - simplified
                static_objects[i, 6:10] = [1, 0, 0, 0]  # Default to first type
        
        old_format['static_objects'] = static_objects
        
        # 6. Lanes and route lanes
        lanes = np.zeros((self._lane_num, self._lane_len, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((self._lane_num, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((self._lane_num, 1), dtype=np.bool_)
        
        route_lanes = np.zeros((self._route_num, self._route_len, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((self._route_num, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((self._route_num, 1), dtype=np.bool_)
        
        # Extract lane information from polygon/polyline data
        if polygon_data and 'type' in polygon_data:
            lane_mask = polygon_data['type'] == 0  # LANE type
            lane_indices = np.where(lane_mask)[0]
            
            # Get polyline data
            polyline_data = data.get('polyline', {})
            polyline_to_polygon = data.get(('polyline', 'polygon'), {}).get('polyline_to_polygon_edge_index', [[], []])
            
            # Process regular lanes
            num_lanes_processed = 0
            num_routes_processed = 0
            
            for poly_idx in lane_indices:
                if 'on_route_mask' in polygon_data and polygon_data['on_route_mask'][poly_idx]:
                    # This is a route lane
                    if num_routes_processed < self._route_num:
                        # Get polylines for this polygon
                        polyline_indices = np.where(polyline_to_polygon[1] == poly_idx)[0]
                        
                        if len(polyline_indices) > 0 and 'position' in polyline_data:
                            # Build lane features
                            for j, pl_idx in enumerate(polyline_indices[:self._route_len]):
                                if pl_idx < len(polyline_data['position']):
                                    # Position and vector
                                    route_lanes[num_routes_processed, j, 0:2] = polyline_data['position'][pl_idx]
                                    # Add other features as available
                                    
                        # Speed limit
                        if polygon_data.get('speed_limit_valid_mask', [False])[poly_idx]:
                            route_lanes_speed_limit[num_routes_processed] = polygon_data['speed_limit'][poly_idx]
                            route_lanes_has_speed_limit[num_routes_processed] = True
                            
                        num_routes_processed += 1
                else:
                    # Regular lane
                    if num_lanes_processed < self._lane_num:
                        # Similar processing for regular lanes
                        polyline_indices = np.where(polyline_to_polygon[1] == poly_idx)[0]
                        
                        if len(polyline_indices) > 0 and 'position' in polyline_data:
                            for j, pl_idx in enumerate(polyline_indices[:self._lane_len]):
                                if pl_idx < len(polyline_data['position']):
                                    lanes[num_lanes_processed, j, 0:2] = polyline_data['position'][pl_idx]
                                    
                        if polygon_data.get('speed_limit_valid_mask', [False])[poly_idx]:
                            lanes_speed_limit[num_lanes_processed] = polygon_data['speed_limit'][poly_idx]
                            lanes_has_speed_limit[num_lanes_processed] = True
                            
                        num_lanes_processed += 1
        
        old_format['lanes'] = lanes
        old_format['lanes_speed_limit'] = lanes_speed_limit
        old_format['lanes_has_speed_limit'] = lanes_has_speed_limit
        old_format['route_lanes'] = route_lanes
        old_format['route_lanes_speed_limit'] = route_lanes_speed_limit
        old_format['route_lanes_has_speed_limit'] = route_lanes_has_speed_limit
        
        return old_format
    
    def __getitem__(self, idx: int) -> Tuple:
        """Load new format data and convert to old format tuple"""
        file_name = self.idx_to_file[idx]
        file_info = self._file_index[file_name]
        
        # Get file path
        file_path = Path(file_info['path'])
        if not file_path.is_absolute() and self.data_dir:
            file_path = self.data_dir / file_path
        
        # Change extension to .pt if needed
        if file_path.suffix == '.npz':
            file_path = file_path.with_suffix('.pt')
        
        # Load PyTorch file with new format
        new_data = torch.load(str(file_path), map_location='cpu')
        
        # Convert to old format
        old_data = self._convert_new_to_old_format(new_data)
        
        # Add metadata
        old_data['scenario_type'] = file_info.get('scenario_type', 'unknown')
        old_data['file_name'] = file_name
        
        # Return as tuple in expected order (matching original __getitem__)
        return tuple([
            old_data['ego_current_state'],
            old_data['ego_agent_future'],
            old_data['neighbor_agents_past'],
            old_data['neighbor_agents_future'],
            old_data['lanes'],
            old_data['lanes_speed_limit'],
            old_data['lanes_has_speed_limit'],
            old_data['route_lanes'],
            old_data['route_lanes_speed_limit'],
            old_data['route_lanes_has_speed_limit'],
            old_data['static_objects'],
            old_data['scenario_type'],
            old_data['file_name']
        ])
    
    # def __getitem__(self, idx: int) -> Tuple:
    #     file_name = self.idx_to_file[idx]
    #     file_info = self._file_index[file_name]
        
    #     # Get file path (use absolute path from index or construct from data_dir)
    #     file_path = Path(file_info['path'])
    #     if not file_path.is_absolute() and self.data_dir:
    #         file_path = self.data_dir / file_path
        
    #     # Load data
    #     data = opendata(str(file_path))
        
    #     # Extract data fields
    #     ego_current_state = data['ego_current_state']
    #     ego_agent_future = data['ego_agent_future'][:self._future_len]
        
    #     neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
    #     neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]
        
    #     lanes = data['lanes']
    #     lanes_speed_limit = data['lanes_speed_limit']
    #     lanes_has_speed_limit = data['lanes_has_speed_limit']
        
    #     route_lanes = data['route_lanes']
    #     route_lanes_speed_limit = data['route_lanes_speed_limit']
    #     route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']
        
    #     static_objects = data['static_objects']
        
    #     # Include scenario type in output if needed
    #     output_data = {
    #         "ego_current_state": ego_current_state,
    #         "ego_future_gt": ego_agent_future,
    #         "neighbor_agents_past": neighbor_agents_past,
    #         "neighbors_future_gt": neighbor_agents_future,
    #         "lanes": lanes,
    #         "lanes_speed_limit": lanes_speed_limit,
    #         "lanes_has_speed_limit": lanes_has_speed_limit,
    #         "route_lanes": route_lanes,
    #         "route_lanes_speed_limit": route_lanes_speed_limit,
    #         "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
    #         "static_objects": static_objects,
    #         "scenario_type": file_info['scenario_type'],  # Add metadata
    #         "file_name": file_name
    #     }
        
    #     return tuple(output_data.values())



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