import os
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData


class NuplanDataset(Dataset):
    def __init__(self,
                 root: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 ) -> None:

        self.root = root
        with open(metadata_path, 'r') as metadata_f:
            metadata = json.load(metadata_f)

        # TODO:
        self._raw_file_names = []

        self._processed_paths = []
        self._processed_file_names = []
        if 'file_index' in metadata:
            for filename, file_info in metadata['file_index'].items():
                self._processed_paths.append(file_info['path'])
                self._processed_file_names.append(filename)

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
        return HeteroData(torch.load(self.processed_paths[idx]))