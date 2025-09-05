import os
import time
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import Deque, Dict, List, Type, Optional

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput, PlannerReport
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from diffusion_planner.model.factorized_diffusion_planner import Factorized_Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

from diffusion_planner.feature_builders.nuplan_scenario_render import NuplanScenarioRender
from diffusion_planner.scenario_manager.scenario_manager import ScenarioManager

def identity(ego_state, predictions):
    return predictions


class FactorizedDiffusionPlanner(AbstractPlanner):
    requires_scenario: bool = True
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            temperature = 0.5,
            diffusion_steps = 10,

            enable_ema: bool = True,
            device: str = "cpu",

            eval_dt: float = 0.1,
            eval_num_frames: int = 80,
            render: bool = True,
            save_dir = None,
            scenario = None,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self.temperature = temperature
        self.diffusion_steps = diffusion_steps

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Factorized_Diffusion_Planner(config)

        self._planner_feature_builder = self._planner.get_list_of_required_feature()[0]
        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        # Add visualization components
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
   
        self._scenario_manager: Optional[ScenarioManager] = None
        self._render = render
        self._imgs = []
        self._scenario = scenario
        if self._render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        
        if self._ckpt_path is not None:
            # Load Lightning checkpoint
            checkpoint = torch.load(self._ckpt_path, map_location=self._device)
            
            # Extract model weights from Lightning state_dict
            lightning_state_dict = checkpoint['state_dict']
            
            if self._ema_enabled:
                # Look for EMA weights (stored with model_ema prefix in Lightning)
                model_state_dict = {}
                for key, value in lightning_state_dict.items():
                    if key.startswith('model_ema.ema.'):
                        new_key = key.replace('model_ema.ema.', '')
                        model_state_dict[new_key] = value
                
                # Fallback to regular model weights if EMA not found
                if not model_state_dict:
                    assert False, print(lightning_state_dict.keys())
                    model_state_dict = {k.replace('model.', ''): v 
                                    for k, v in lightning_state_dict.items() 
                                    if k.startswith('model.')}
            else:
                # Use regular model weights
                model_state_dict = {k.replace('model.', ''): v 
                                for k, v in lightning_state_dict.items() 
                                if k.startswith('model.')}
            
            # Load weights into model
            self._planner.load_state_dict(model_state_dict, strict=False)
            print(f"Loaded checkpoint from: {self._ckpt_path}")
        else:
            print("No checkpoint provided, using random weights")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=self._map_api,
            ego_state=None,
            route_roadblocks_ids=self._route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager
        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        # """
        # Inherited.
        # """
        # inputs = self.planner_input_to_model_inputs(current_input)

        # inputs = self.observation_normalizer(inputs)        
        # _, outputs = self._planner(inputs)

        # trajectory = InterpolatedTrajectory(
        #     trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        # )

        # return trajectory
        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()
        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        trajectory = self._run_planning_once(current_input)
        self._inference_runtimes.append(time.perf_counter() - start_time)
        return trajectory

    def _run_planning_once(self, current_input: PlannerInput):
        inputs = self.planner_input_to_model_inputs(current_input)
        inputs = self.observation_normalizer(inputs)        
        start = time.time()
        inputs['temperature'] = self.temperature
        inputs['diffusion_steps'] = self.diffusion_steps 
        _, outputs = self._planner(inputs)
        end = time.time()
        print(f"diffusion cost time: {end - start}")

        ego_state = current_input.history.ego_states[-1] # (?)

        output_trajectories = outputs['prediction'][:, 0, :, :].detach().cpu().numpy().astype(np.float64)  # [B, P, T, 4] -> [B, T, 4]
        candidate_trajectories = self.output_trajectories_process(output_trajectories, ego_state)   # [B, T, 4] -> [B, T + 1, 3]

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        if self._render:
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=self._global_to_local(trajectory, ego_state),
                    candidate_trajectories=self._global_to_local(
                        candidate_trajectories[:], ego_state
                    ),
                    candidate_index=0,
                    predictions=None,
                    return_img=True,
                )
            )

        return trajectory
    
    def output_trajectories_process(self, output_trajectories, ego_state):
        """
        output_trajectories: [N, T, 4] local x, y, cos, sin

        results:
        candidate_trajectories: [N, T, 3] global x y heading
        """
        heading = np.arctan2(output_trajectories[..., 3], output_trajectories[..., 2])[..., None]
        output_trajectories = np.concatenate([output_trajectories[..., :2], heading], axis=-1)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        output_trajectories[..., :2] = (
            np.matmul(output_trajectories[..., :2], rot_mat) + origin
        )
        output_trajectories[..., 2] += angle

        output_trajectories = np.concatenate(
            [output_trajectories[..., 0:1, :], output_trajectories],
            axis=-2,
        )

        return output_trajectories

    def _global_to_local(self, global_trajectory, ego_state):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True):
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio

            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report