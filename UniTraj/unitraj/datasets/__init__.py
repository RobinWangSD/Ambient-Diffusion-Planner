from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .SMART_dataset import SMARTDataset
from .flow_dataset import FlowDataset
from .flow_planner_dataset import FlowPlannerDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    'SMART': SMARTDataset,
    'flow-ego-centric': FlowDataset,
    'flow-planner': FlowPlannerDataset,
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
