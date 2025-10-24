import os
from glob import glob
from typing import Callable, List, Optional, Tuple, Union
import random
from joblib import Parallel, delayed
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from datasets import get_scenario_map
from datasets import get_filter_parameters
from datasets import get_features
from datasets import get_plan_scenario_types


class NuplanDataset(Dataset):
    def __init__(self,
                 root: Optional[str] = None,
                 metadata: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 ) -> None:

        self.root = root
        self.metadata = 

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