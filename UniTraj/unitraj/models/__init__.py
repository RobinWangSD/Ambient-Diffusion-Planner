from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.smart.smart import SMART
from unitraj.models.flow.flow import EgoCentricFlow
from unitraj.models.flow_planner.flow_planner import FlowPlanner

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'SMART': SMART,
    'flow-ego-centric': EgoCentricFlow,
    'flow-planner': FlowPlanner,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
