import os
import pickle
import json
from pathlib import Path
from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Dict, List, Tuple, Optional, Iterator
import random
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from diffusion_planner.utils.train_utils import opendata


class DiffusionPlannerDataDistributed(Dataset):
    """DataLoader optimized for distributed training with mapping file or metadata index."""

    DEFAULT_TRAIN_DIRS = ['train_boston', 'train_pittsburgh', 'train_singapore'] + \
        [f'train_vegas_{i}' for i in range(1, 7)]
    DEFAULT_VAL_DIRS = ['val']
    
    def __init__(
        self, 
        data_dir: Optional[str], 
        mapping_pkl: Optional[str],
        past_neighbor_num: int, 
        predicted_neighbor_num: int, 
        future_len: int,
        filter_prefix: Optional[str] = None,
        max_files: Optional[int] = None,
        metadata_path: Optional[str] = None,
        data_split: str = "train",
        allowed_dirs: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        self._data_split = data_split
        self._allowed_dirs = allowed_dirs
        self._corrupted_indices = set()
        # Defaults match the original Diffusion Planner data layout
        self._static_num = 5
        self._lane_num = 70
        self._lane_len = 20
        self._route_num = 25
        self._route_len = 20
        self._plot_states_target = os.environ.get("DP_PLOT_VALID_STATES", "")
        self._plot_counter = 0
        
        if metadata_path:
            self._mapping = self._load_from_metadata(metadata_path, max_files)
        else:
            if mapping_pkl is None:
                raise ValueError("Either mapping_pkl or metadata_path must be provided.")
            self._mapping = self._load_from_mapping(mapping_pkl, filter_prefix, max_files)
        
        # Create file list
        self.file_list = list(self._mapping.keys())
        self.idx_to_file = {idx: file for idx, file in enumerate(self.file_list)}

        print(f"Dataset initialized with {len(self.file_list):,} files")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple:
        total = len(self.file_list)
        if total == 0:
            raise RuntimeError("No files available to load in DiffusionPlannerDataDistributed.")

        start_idx = idx % total
        attempts = 0
        cur_idx = start_idx

        while attempts < total:
            file_name = self.idx_to_file[cur_idx]
            file_path = self._resolve_file_path(self._mapping[file_name])
            
            try:
                data = self._load_sample(file_path)
                return tuple(data.values())
            except Exception as exc:  # pylint: disable=broad-except
                self._corrupted_indices.add(cur_idx)
                print(f"[DiffusionPlannerDataDistributed] Failed to load {file_path}: {exc}")
                attempts += 1
                cur_idx = (cur_idx + 1) % total

        raise RuntimeError("All processed samples failed to load in DiffusionPlannerDataDistributed.")

    def _default_dirs(self) -> List[str]:
        if self._allowed_dirs:
            return self._allowed_dirs
        if self._data_split == "val":
            return self.DEFAULT_VAL_DIRS
        return self.DEFAULT_TRAIN_DIRS

    def _load_from_mapping(
        self, mapping_pkl: str, filter_prefix: Optional[str], max_files: Optional[int]
    ) -> Dict[str, str]:
        with open(mapping_pkl, 'rb') as f:
            full_mapping = pickle.load(f)

        if filter_prefix:
            mapping = {k: v for k, v in full_mapping.items() if k.startswith(filter_prefix)}
        else:
            mapping = full_mapping

        if max_files and len(mapping) > max_files:
            mapping = dict(list(mapping.items())[:max_files])

        self._full_mapping = full_mapping
        return mapping

    def _load_from_metadata(self, metadata_path: str, max_files: Optional[int]) -> Dict[str, str]:
        with open(metadata_path, 'r') as metadata_f:
            metadata = json.load(metadata_f)

        file_index = metadata.get('file_index', {})
        if not file_index:
            raise ValueError(f"No file_index found in metadata file: {metadata_path}")

        allowed_dirs_list = self._default_dirs()
        allowed_dirs = set(allowed_dirs_list) if allowed_dirs_list else None
        mapping = {}
        for filename, file_info in file_index.items():
            dir_name = file_info.get('dir')
            if filename in ['pre_transform.pt', 'pre_filter.pt']:
                continue
            if allowed_dirs and dir_name not in allowed_dirs:
                continue

            file_path = file_info.get('path')
            if not file_path:
                continue

            mapping[filename] = file_path

            if max_files and len(mapping) >= max_files:
                break

        if not mapping:
            raise ValueError(f"No files matched split '{self._data_split}' using metadata {metadata_path}")

        self._full_mapping = mapping
        return mapping

    def _resolve_file_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if path.is_absolute():
            return path
        if self.data_dir is None:
            raise ValueError("Relative data paths require data_dir to be set.")
        return self.data_dir / path

    def _load_sample(self, file_path: Path) -> Dict[str, np.ndarray]:
        # Support both legacy npz cache and new .pt hetero-graph cache
        if file_path.suffix == ".pt":
            return self._load_from_pt(file_path)

        raw = opendata(str(file_path))
        return {
            "ego_current_state": raw['ego_current_state'],
            "ego_future_gt": raw['ego_agent_future'],
            "neighbor_agents_past": raw['neighbor_agents_past'][:self._past_neighbor_num],
            "neighbors_future_gt": raw['neighbor_agents_future'][:self._predicted_neighbor_num],
            "lanes": raw['lanes'],
            "lanes_speed_limit": raw['lanes_speed_limit'],
            "lanes_has_speed_limit": raw['lanes_has_speed_limit'],
            "route_lanes": raw['route_lanes'],
            "route_lanes_speed_limit": raw['route_lanes_speed_limit'],
            "route_lanes_has_speed_limit": raw['route_lanes_has_speed_limit'],
            "static_objects": raw['static_objects'],
        }

    def _rotate(self, tensor: torch.Tensor, cos_h: torch.Tensor, sin_h: torch.Tensor) -> torch.Tensor:
        """Rotate (x, y) vectors by -heading (cos_h/sin_h precomputed)."""
        x, y = tensor[..., 0], tensor[..., 1]
        rot_x = x * cos_h - y * sin_h
        rot_y = x * sin_h + y * cos_h
        return torch.stack([rot_x, rot_y], dim=-1)

    def _prep_agent_blocks(
        self,
        agent_data: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Convert hetero agent features to Diffusion Planner format.
        Returns:
            ego_current_state: (10,)
            ego_future_gt: (future_len, 3)
            neighbor_agents_past: (past_neighbor_num, time_len, 11)
            neighbor_agents_future: (predicted_neighbor_num, future_len, 3)
            rotation: (cos_h, sin_h, anchor_heading)
        """
        position = agent_data['position'].float()          # [N, T, 2], centered on ego position
        heading = agent_data['heading'].float()            # [N, T], global orientation
        velocity = agent_data['velocity'].float()          # [N, T, 2], global frame (ego velocity already rotated)
        visible_mask = agent_data['visible_mask'].bool()   # [N, T]
        agent_type = agent_data['type'].long()             # [N]
        agent_identity = agent_data['identity'].long()     # [N], 0 for ego
        agent_box = agent_data['box'].float()              # [N, 4] (front, back, left, right half-lengths)

        total_steps = position.shape[1]
        past_len = total_steps - self._future_len
        current_idx = past_len - 1

        anchor_heading = heading[0, current_idx]
        cos_h = torch.cos(-anchor_heading)
        sin_h = torch.sin(-anchor_heading)

        pos_rot = self._rotate(position, cos_h, sin_h)
        vel_rot = velocity.clone()
        if vel_rot.shape[0] > 0:
            non_ego_mask = agent_identity != 0
            vel_rot[non_ego_mask] = self._rotate(velocity[non_ego_mask], cos_h, sin_h)
        heading_rel = heading - anchor_heading
        heading_rel = (heading_rel + math.pi) % (2 * math.pi) - math.pi

        # Ego current state
        ego_pos = pos_rot[0, current_idx]
        ego_heading_cos = torch.cos(heading_rel[0, current_idx])
        ego_heading_sin = torch.sin(heading_rel[0, current_idx])
        ego_vel = vel_rot[0, current_idx]
        dt = 0.1
        if current_idx > 0:
            prev_heading = heading_rel[0, current_idx - 1]
            yaw_rate = (heading_rel[0, current_idx] - prev_heading) / dt
            prev_vel = vel_rot[0, current_idx - 1]
            acc = (ego_vel - prev_vel) / dt
        else:
            yaw_rate = vel_rot.new_tensor(0.0)
            if vel_rot.shape[1] > 1:
                acc = (vel_rot[0, 1] - ego_vel) / dt
            else:
                acc = vel_rot.new_zeros(2)

        ego_current_state = torch.tensor([
            ego_pos[0], ego_pos[1],
            ego_heading_cos, ego_heading_sin,
            ego_vel[0], ego_vel[1],
            acc[0], acc[1],
            0.0,             # steering angle placeholder
            yaw_rate,
        ], dtype=torch.float32)

        # Ego future gt: x, y, heading (relative)
        ego_future_pos = pos_rot[0, past_len:]
        ego_future_heading = heading_rel[0, past_len:]
        ego_future_gt = torch.cat([ego_future_pos, ego_future_heading.unsqueeze(-1)], dim=-1)

        # Neighbor ordering by distance at current step; skip ego
        non_ego_mask = agent_identity != 0
        non_ego_indices = torch.nonzero(non_ego_mask, as_tuple=False).squeeze(-1)

        if non_ego_indices.numel() > 0:
            current_pos = pos_rot[non_ego_mask, current_idx]  # [M, 2]
            current_visible = visible_mask[non_ego_mask, current_idx]
            distances = torch.where(current_visible, torch.norm(current_pos, dim=-1), torch.full_like(current_pos[:, 0], float('inf')))
            sorted_indices = torch.argsort(distances)
            ordered_indices = non_ego_indices[sorted_indices]
        else:
            ordered_indices = torch.tensor([], dtype=torch.long)

        # Neighbor past
        neighbor_past = torch.zeros((self._past_neighbor_num, past_len, 11), dtype=torch.float32)
        neighbor_future = torch.zeros((self._predicted_neighbor_num, self._future_len, 3), dtype=torch.float32)

        for slot, agent_idx in enumerate(ordered_indices[:self._past_neighbor_num]):
            pos_hist = pos_rot[agent_idx, :past_len]
            vel_hist = vel_rot[agent_idx, :past_len]
            heading_hist = heading_rel[agent_idx, :past_len]

            agent_type_one_hot = F.one_hot(agent_type[agent_idx], num_classes=3).float()
            width = agent_box[agent_idx, 2] + agent_box[agent_idx, 3]
            length = agent_box[agent_idx, 0] + agent_box[agent_idx, 1]

            neighbor_past[slot, :, 0:2] = pos_hist
            neighbor_past[slot, :, 2] = torch.cos(heading_hist)
            neighbor_past[slot, :, 3] = torch.sin(heading_hist)
            neighbor_past[slot, :, 4:6] = vel_hist
            neighbor_past[slot, :, 6] = width
            neighbor_past[slot, :, 7] = length
            neighbor_past[slot, :, 8:] = agent_type_one_hot

        for slot, agent_idx in enumerate(ordered_indices[:self._predicted_neighbor_num]):
            fut_pos = pos_rot[agent_idx, past_len:]
            fut_heading = heading_rel[agent_idx, past_len:]
            neighbor_future[slot, :, 0:2] = fut_pos
            neighbor_future[slot, :, 2] = fut_heading

        return (
            ego_current_state,
            ego_future_gt,
            neighbor_past,
            neighbor_future,
            (cos_h, sin_h, anchor_heading),
        )

    def _plot_processed_states(
        self,
        ego_current_state: torch.Tensor,
        ego_future_gt: torch.Tensor,
        neighbor_past: torch.Tensor,
        neighbor_future: torch.Tensor,
        lanes: torch.Tensor,
        route_lanes: torch.Tensor,
        static_objects: torch.Tensor,
    ) -> None:
        """Plot processed agent states and map context for quick verification."""
        if not self._plot_states_target:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Road features
        lane_label_added = False
        lane_coords = lanes[:, :, 0:2]
        valid_lanes = lane_coords.abs().sum(dim=(1, 2)) > 0
        for lane in lane_coords[valid_lanes]:
            coords = lane.cpu().numpy()
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="#b0b0b0",
                linewidth=1.0,
                alpha=0.5,
                zorder=1,
                label="lane" if not lane_label_added else None,
            )
            lane_label_added = True

        route_label_added = False
        route_coords = route_lanes[:, :, 0:2]
        valid_routes = route_coords.abs().sum(dim=(1, 2)) > 0
        for route in route_coords[valid_routes]:
            coords = route.cpu().numpy()
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                color="#ff8c00",
                linewidth=1.8,
                alpha=0.8,
                zorder=2,
                label="route lane" if not route_label_added else None,
            )
            route_label_added = True

        static_label_added = False
        static_mask = static_objects.abs().sum(dim=1) > 0
        for obj in static_objects[static_mask].cpu().numpy():
            center = obj[0:2]
            heading = math.atan2(obj[3], obj[2]) if obj[2] or obj[3] else 0.0
            width = obj[4]
            length = obj[5]
            d = math.hypot(length, width)
            theta_2 = math.atan2(width, length)
            pivot_x = center[0] - (d / 2) * math.cos(heading + theta_2)
            pivot_y = center[1] - (d / 2) * math.sin(heading + theta_2)
            rect = Rectangle(
                xy=(pivot_x, pivot_y),
                width=length,
                height=width,
                angle=math.degrees(heading),
                fc="crimson",
                ec="darkred",
                alpha=0.45,
                zorder=30,
                label="static object" if not static_label_added else None,
            )
            ax.add_patch(rect)
            static_label_added = True

        ego_color = "tab:blue"
        neighbor_cmap = plt.cm.get_cmap("tab10")

        ego_future = ego_future_gt[..., :2].cpu().numpy()
        ego_cur = ego_current_state[:2].cpu().numpy()
        ax.scatter(ego_cur[0], ego_cur[1], color=ego_color, marker="o", zorder=5, label="ego current")
        ax.plot(ego_future[:, 0], ego_future[:, 1], color=ego_color, linestyle="--", linewidth=2, label="ego future")

        past_label_added = False
        future_label_added = False

        for slot in range(neighbor_past.shape[0]):
            agent_color = neighbor_cmap(slot % 10)
            past_xy = neighbor_past[slot, :, 0:2]
            past_mask = past_xy.abs().sum(dim=-1) > 0
            if past_mask.any():
                traj = past_xy[past_mask].cpu().numpy()
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=agent_color,
                    linewidth=1.5,
                    alpha=0.85,
                    label="neighbor past" if not past_label_added else None,
                )
                past_label_added = True

        for slot in range(neighbor_future.shape[0]):
            agent_color = neighbor_cmap(slot % 10)
            fut_xy = neighbor_future[slot, :, 0:2]
            fut_mask = fut_xy.abs().sum(dim=-1) > 0
            if fut_mask.any():
                traj = fut_xy[fut_mask].cpu().numpy()
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=agent_color,
                    linestyle="--",
                    linewidth=1.25,
                    alpha=0.7,
                    label="neighbor future" if not future_label_added else None,
                )
                future_label_added = True

        ax.legend(loc="upper right", frameon=False)
        ax.set_title("Processed scene (ego-centered)")

        target = self._plot_states_target.lower()
        if target in {"1", "true", "yes", "show"}:
            plt.show(block=False)
            plt.pause(0.001)
        else:
            output_dir = Path(self._plot_states_target)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"state_{self._plot_counter:06d}.png", dpi=200, bbox_inches="tight")
            self._plot_counter += 1
        plt.close(fig)

    def _resample_polyline(self, points: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resample polyline to target_len points using linear interpolation."""
        if points.shape[0] == target_len:
            return points
        t_orig = np.linspace(0.0, 1.0, num=points.shape[0], dtype=np.float32)
        t_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
        x = np.interp(t_new, t_orig, points[:, 0].cpu().numpy())
        y = np.interp(t_new, t_orig, points[:, 1].cpu().numpy())
        return torch.from_numpy(np.stack([x, y], axis=-1)).to(points.dtype)

    def _traffic_one_hot(self, status: torch.Tensor) -> torch.Tensor:
        """Map traffic light status (0-4) to one-hot of length 4 (G,Y,R,UNK)."""
        one_hot = torch.zeros(4, dtype=torch.float32)
        if status == 0:
            one_hot[0] = 1.0
        elif status == 1:
            one_hot[1] = 1.0
        elif status == 2:
            one_hot[2] = 1.0
        else:
            one_hot[3] = 1.0
        return one_hot

    def _prep_static_and_map(
        self,
        raw: Dict[str, torch.Tensor],
        rotation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert polygon/polyline data into DP lane/static tensors."""
        cos_h, sin_h, anchor_heading = rotation

        polygon = raw['polygon']
        polyline = raw['polyline']
        edge_map = raw[('polyline', 'polygon')]

        polygon_position = polygon['position'].float()          # [P, 2], ego-centered (translation)
        polygon_heading = polygon['heading'].float()            # [P]
        polygon_type = polygon['type'].long()                   # [P]
        polygon_speed_limit = polygon['speed_limit'].float()    # [P]
        polygon_speed_limit_valid = polygon['speed_limit_valid_mask'].bool()  # [P]
        polygon_traffic_light = polygon['traffic_light'].long() # [P]
        polygon_on_route = polygon['on_route_mask'].bool()      # [P]

        poly_pos = polyline['position'].float()                 # [S, 2]
        poly_heading = polyline['heading'].float()              # [S]
        poly_length = polyline['length'].float()                # [S]
        poly_to_polygon = edge_map['polyline_to_polygon_edge_index'].long()  # [2, S]

        num_polygons = polygon_type.shape[0]
        counts = torch.bincount(poly_to_polygon[1], minlength=num_polygons)
        offsets = torch.cumsum(torch.cat([torch.tensor([0]), counts[:-1]]), dim=0)

        # Rotate polygon positions to ego heading
        poly_pos_rot = self._rotate(poly_pos, cos_h, sin_h)
        polygon_pos_rot = self._rotate(polygon_position, cos_h, sin_h)
        poly_heading_rel = poly_heading - anchor_heading
        poly_heading_rel = (poly_heading_rel + math.pi) % (2 * math.pi) - math.pi
        polygon_heading_rel = polygon_heading - anchor_heading
        polygon_heading_rel = (polygon_heading_rel + math.pi) % (2 * math.pi) - math.pi

        # Lanes
        lane_mask = polygon_type == 0  # LANE
        lane_indices = torch.nonzero(lane_mask, as_tuple=False).squeeze(-1)
        # sort by distance to ego
        lane_dists = torch.norm(polygon_pos_rot[lane_indices], dim=-1) if lane_indices.numel() > 0 else torch.tensor([])
        if lane_indices.numel() > 0:
            sorted_idx = torch.argsort(lane_dists)
            lane_indices = lane_indices[sorted_idx]
        lane_indices = lane_indices[:self._lane_num]

        lanes = torch.zeros((self._lane_num, self._lane_len, 12), dtype=torch.float32)
        lanes_speed_limit = torch.zeros((self._lane_num, 1), dtype=torch.float32)
        lanes_has_speed_limit = torch.zeros((self._lane_num, 1), dtype=torch.bool)

        route_lanes = torch.zeros((self._route_num, self._route_len, 12), dtype=torch.float32)
        route_lanes_speed_limit = torch.zeros((self._route_num, 1), dtype=torch.float32)
        route_lanes_has_speed_limit = torch.zeros((self._route_num, 1), dtype=torch.bool)

        route_slot = 0
        for slot, poly_idx in enumerate(lane_indices):
            num_seg = counts[poly_idx].item()
            if num_seg == 0:
                continue
            start = offsets[poly_idx].item()
            end = start + num_seg
            seg_pos = poly_pos_rot[start:end]
            seg_heading = poly_heading_rel[start:end]
            seg_len = poly_length[start:end]

            # reconstruct points
            pts = torch.zeros((num_seg + 1, 2), dtype=torch.float32)
            pts[:-1] = seg_pos
            vec = seg_len.unsqueeze(-1) * torch.stack([torch.cos(seg_heading), torch.sin(seg_heading)], dim=-1)
            pts[1:] = seg_pos + vec

            pts = self._resample_polyline(pts, self._lane_len)
            vectors = torch.zeros_like(pts)
            vectors[:-1] = pts[1:] - pts[:-1]

            traffic = self._traffic_one_hot(polygon_traffic_light[poly_idx])
            traffic = traffic.unsqueeze(0).repeat(self._lane_len, 1)

            lane_feat = torch.zeros((self._lane_len, 12), dtype=torch.float32)
            lane_feat[:, 0:2] = pts
            lane_feat[:, 2:4] = vectors
            # left/right offsets not available -> zeros
            lane_feat[:, 8:12] = traffic
            lanes[slot] = lane_feat

            lanes_speed_limit[slot, 0] = polygon_speed_limit[poly_idx]
            lanes_has_speed_limit[slot, 0] = polygon_speed_limit_valid[poly_idx]

            if polygon_on_route[poly_idx] and route_slot < self._route_num:
                route_lanes[route_slot] = lane_feat
                route_lanes_speed_limit[route_slot, 0] = polygon_speed_limit[poly_idx]
                route_lanes_has_speed_limit[route_slot, 0] = polygon_speed_limit_valid[poly_idx]
                route_slot += 1

        # Static objects
        static_mask = polygon_type == 3  # STATIC_OBJECT
        static_indices = torch.nonzero(static_mask, as_tuple=False).squeeze(-1)
        static_objects = torch.zeros((self._static_num, 10), dtype=torch.float32)
        if static_indices.numel() > 0:
            static_dists = torch.norm(polygon_pos_rot[static_indices], dim=-1)
            sorted_static = static_indices[torch.argsort(static_dists)][:self._static_num]
            for slot, poly_idx in enumerate(sorted_static):
                num_seg = counts[poly_idx].item()
                start = offsets[poly_idx].item()
                end = start + num_seg
                seg_pos = poly_pos_rot[start:end]
                seg_heading = poly_heading_rel[start:end]
                seg_len = poly_length[start:end]
                pts = torch.zeros((num_seg + 1, 2), dtype=torch.float32)
                pts[:-1] = seg_pos
                vec = seg_len.unsqueeze(-1) * torch.stack([torch.cos(seg_heading), torch.sin(seg_heading)], dim=-1)
                pts[1:] = seg_pos + vec

                min_xy, _ = torch.min(pts, dim=0)
                max_xy, _ = torch.max(pts, dim=0)
                width = (max_xy[1] - min_xy[1]).abs()
                length = (max_xy[0] - min_xy[0]).abs()

                heading_rel = polygon_heading_rel[poly_idx]
                static_objects[slot, 0:2] = polygon_pos_rot[poly_idx]
                static_objects[slot, 2] = torch.cos(heading_rel)
                static_objects[slot, 3] = torch.sin(heading_rel)
                static_objects[slot, 4] = width
                static_objects[slot, 5] = length
                # type one-hot: use last channel as generic
                static_objects[slot, 6:] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

        return (
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,
            static_objects,
        )

    def _load_from_pt(self, file_path: Path) -> Dict[str, torch.Tensor]:
        """Load new .pt cache and convert to Diffusion Planner-compatible dict."""
        raw = torch.load(str(file_path), map_location="cpu")
        plot_enabled = bool(self._plot_states_target)
        (
            ego_current_state,
            ego_future_gt,
            neighbor_agents_past,
            neighbor_agents_future,
            rotation,
        ) = self._prep_agent_blocks(raw['agent'])

        (
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,
            static_objects,
        ) = self._prep_static_and_map(raw, rotation)

        if plot_enabled:
            self._plot_processed_states(
                ego_current_state,
                ego_future_gt,
                neighbor_agents_past,
                neighbor_agents_future,
                lanes,
                route_lanes,
                static_objects,
            )

        data = {
            "ego_current_state": ego_current_state.numpy(),
            "ego_future_gt": ego_future_gt.numpy(),
            "neighbor_agents_past": neighbor_agents_past.numpy(),
            "neighbors_future_gt": neighbor_agents_future.numpy(),
            "lanes": lanes.numpy(),
            "lanes_speed_limit": lanes_speed_limit.numpy(),
            "lanes_has_speed_limit": lanes_has_speed_limit.numpy(),
            "route_lanes": route_lanes.numpy(),
            "route_lanes_speed_limit": route_lanes_speed_limit.numpy(),
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit.numpy(),
            "static_objects": static_objects.numpy(),
        }

        return data


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
