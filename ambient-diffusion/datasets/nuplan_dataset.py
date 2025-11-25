import os
import json
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData


class NuplanDataset(Dataset):
    def __init__(self,
                 root: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 data_type: str = 'train',
                 ) -> None:

        self.root = root
        with open(metadata_path, 'r') as metadata_f:
            metadata = json.load(metadata_f)

        if data_type == 'train':
            self.dirs = ['train_boston', 'train_pittsburgh', 'train_singapore', 'train_vegas_1', 'train_vegas_2', 'train_vegas_3', 'train_vegas_4', 'train_vegas_5', 'train_vegas_6']
        elif data_type == 'val':
            self.dirs = ['val']

        self._raw_file_names = []

        self._processed_paths = []
        self._processed_file_names = []
        if 'file_index' in metadata:
            for filename, file_info in metadata['file_index'].items():
                if file_info['dir'] in self.dirs:
                    self._processed_paths.append(file_info['path'])
                    self._processed_file_names.append(filename)

        self._corrupted_indices = set()

        super(NuplanDataset, self).__init__(root=root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'nuplan-v1.1', 'splits', self.dir)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'nuplan-v1.1', 'processed', f"{self.dir}-processed-{self.mode}-PlanR1")
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
        
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:     
        total = len(self.processed_paths)
        if total == 0:
            raise RuntimeError("No processed paths available to load.")

        start_idx = idx % total
        if start_idx in self._corrupted_indices:
            start_idx = (start_idx + 1) % total

        attempts = 0
        cur_idx = start_idx
        while attempts < total:
            path = self.processed_paths[cur_idx]
            try:
                return HeteroData(torch.load(path))
            except Exception as exc:  # pylint: disable=broad-except
                self._corrupted_indices.add(cur_idx)
                print(f"[NuplanDataset] Failed to load {path}: {exc}")
                attempts += 1
                cur_idx = (cur_idx + 1) % total

        raise RuntimeError("All processed samples failed to load in NuplanDataset.")
