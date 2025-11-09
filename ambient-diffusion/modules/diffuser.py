import math
from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse

from layers import GraphAttention, TwoLayerMLP
from utils import (
    compute_angles_lengths_2D,
    transform_point_to_local_coordinate,
    wrap_angle,
    drop_edge_between_samples,
)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding followed by a linear projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class Diffuser(nn.Module):
    """
    Diffusion denoiser that treats each agent-timestep pair as a node and
    performs attention over (past -> future), (future <-> map), and (future <-> future).
    """

    model_type = "eps"  # used by dpm-solver wrapper

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 128,
        num_future_steps: int = 80,
        num_historical_steps: int = 20,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        temporal_span: int = 6,
        agent_radius: float = 60.0,
        polygon_radius: float = 30.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.num_historical_steps = num_historical_steps
        self.temporal_span = temporal_span
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius

        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.future_embed = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.past_future_edge_mlp = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.map_future_edge_mlp = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.future_future_edge_mlp = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.past_future_attn = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.map_future_attn = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.future_self_attn = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        data: Batch,
        map_embeddings: torch.Tensor,
        past_embs: torch.Tensor,
        past_mask: torch.Tensor,
        x_t: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            data: hetero batch with agent/map attributes.
            map_embeddings: [num_map_nodes, hidden_dim]
            past_embs: [num_agents, num_historical_steps, hidden_dim]
            past_mask: bool [num_agents, num_historical_steps]
            x_t: noisy future states [num_agents, num_future_steps, state_dim]
            diffusion_time: [num_agents] or [num_agents, 1]

        Returns:
            Predicted noise with same shape as x_t.
        """
        if x_t.numel() == 0:
            return torch.zeros_like(x_t)

        device = x_t.device
        num_agents, num_future, _ = x_t.shape
        past_steps = min(self.num_historical_steps, past_embs.size(1))
        past_embs = past_embs[:, :past_steps]
        past_mask = past_mask[:, :past_steps]

        diffusion_time = diffusion_time.reshape(num_agents, -1)[:, 0]
        t_emb = self.time_embed(diffusion_time).unsqueeze(1).expand(-1, num_future, -1)
        future_emb = self.future_embed(torch.cat([x_t, t_emb], dim=-1))

        future_mask = data['agent'].get(
            'target_valid_mask',
            torch.ones((num_agents, num_future), device=device),
        )
        future_mask = future_mask[:, :num_future].to(dtype=torch.bool)

        past_pos = data['agent']['position'][:, :past_steps, :2]
        past_heading = data['agent']['heading'][:, :past_steps]

        future_position = self._get_future_position(data, x_t)
        future_heading = self._get_future_heading(data, num_future, past_heading[:, -1])

        batch_agents = data['agent'].get(
            'batch',
            torch.zeros(num_agents, device=device, dtype=torch.long),
        )
        map_batch = data['polygon'].get(
            'batch',
            torch.zeros(map_embeddings.size(0), device=device, dtype=torch.long),
        )
        map_heading = data['polygon'].get(
            'heading',
            torch.zeros(map_embeddings.size(0), device=device, dtype=future_heading.dtype),
        )
        map_heading_valid = data['polygon'].get(
            'heading_valid_mask',
            torch.ones(map_embeddings.size(0), device=device, dtype=future_heading.dtype),
        )
        map_pos = data['polygon']['position'][:, :2]

        future_flat = future_emb.reshape(-1, self.hidden_dim)
        past_flat = past_embs.reshape(-1, self.hidden_dim)

        edges_pf = self._build_past_future_edges(
            past_pos,
            past_heading,
            past_mask,
            future_position,
            future_heading,
            future_mask,
        )
        edges_mf = self._build_map_future_edges(
            map_pos,
            map_heading,
            map_heading_valid,
            map_batch,
            future_position,
            future_heading,
            future_mask,
            batch_agents,
        )
        edges_ff = self._build_future_future_edges(
            future_position,
            future_heading,
            future_mask,
            batch_agents,
        )

        for i in range(len(self.past_future_attn)):
            future_flat = self.past_future_attn[i](
                x=[past_flat, future_flat],
                edge_index=edges_pf[0],
                edge_attr=edges_pf[1],
            )
            future_flat = self.map_future_attn[i](
                x=[map_embeddings, future_flat],
                edge_index=edges_mf[0],
                edge_attr=edges_mf[1],
            )
            future_flat = self.future_self_attn[i](
                x=future_flat,
                edge_index=edges_ff[0],
                edge_attr=edges_ff[1],
            )

        future_emb = future_flat.view(num_agents, num_future, self.hidden_dim)
        pred_noise = self.output_head(future_emb)
        return pred_noise

    def _get_future_position(self, data: Batch, x_t: torch.Tensor) -> torch.Tensor:
        future_pos = data['agent'].get('future_position', None)
        if future_pos is None:
            target = data['agent'].get('target', None)
            if target is not None and target.size(-1) >= 2:
                future_pos = target[..., :2]
            else:
                future_pos = x_t[..., :2]
        return future_pos[:, :x_t.size(1)]

    def _get_future_heading(self, data: Batch, num_future: int, default_heading: torch.Tensor) -> torch.Tensor:
        heading = data['agent'].get('future_heading', None)
        if heading is None:
            heading = data['agent'].get('target_heading', None)
        if heading is None:
            heading = default_heading.unsqueeze(1).expand(-1, num_future)
        return heading[:, :num_future]

    def _build_past_future_edges(
        self,
        past_pos: torch.Tensor,
        past_heading: torch.Tensor,
        past_mask: torch.Tensor,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = past_pos.device
        num_agents, past_steps, _ = past_pos.shape
        future_steps = future_pos.size(1)

        if num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        hist_idx = torch.arange(num_agents * past_steps, device=device).view(num_agents, past_steps)
        future_idx = torch.arange(num_agents * future_steps, device=device).view(num_agents, future_steps)
        connectivity = past_mask.unsqueeze(-1) & future_mask.unsqueeze(1)

        if not connectivity.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        src = hist_idx.unsqueeze(-1).expand(-1, past_steps, future_steps)[connectivity]
        dst = future_idx.unsqueeze(1).expand(-1, past_steps, future_steps)[connectivity]

        gather_mask = connectivity
        rel_vector = transform_point_to_local_coordinate(
            past_pos.unsqueeze(-2).expand(-1, -1, future_steps, -1)[gather_mask],
            future_pos.unsqueeze(1).expand(-1, past_steps, -1, -1)[gather_mask],
            future_heading.unsqueeze(1).expand(-1, past_steps, -1)[gather_mask],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(
            past_heading.unsqueeze(-1).expand(-1, -1, future_steps)[gather_mask]
            - future_heading.unsqueeze(1).expand(-1, past_steps, -1)[gather_mask]
        )

        time_hist = torch.arange(past_steps, device=device).view(1, past_steps, 1)
        time_future = torch.arange(future_steps, device=device).view(1, 1, future_steps)
        delta_t = (time_future - time_hist).expand(num_agents, -1, -1)[gather_mask]

        edge_features = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
                delta_t.float(),
            ],
            dim=-1,
        )
        edge_attr = self.past_future_edge_mlp(edge_features)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, edge_attr

    def _build_map_future_edges(
        self,
        map_pos: torch.Tensor,
        map_heading: torch.Tensor,
        map_heading_valid: torch.Tensor,
        map_batch: torch.Tensor,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_mask: torch.Tensor,
        agent_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = map_pos.device
        num_map = map_pos.size(0)
        num_agents, future_steps, _ = future_pos.shape
        if num_map == 0 or num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        future_flat = future_pos.reshape(-1, 2)
        heading_flat = future_heading.reshape(-1)
        mask_flat = future_mask.reshape(-1)
        future_batch = agent_batch.unsqueeze(1).expand(-1, future_steps).reshape(-1)

        valid = torch.ones((num_map, future_flat.size(0)), dtype=torch.bool, device=device)
        valid = drop_edge_between_samples(valid, batch=(map_batch, future_batch))
        valid = valid & mask_flat.unsqueeze(0)

        dist = torch.cdist(map_pos[:, :2], future_flat[:, :2])
        valid = valid & (dist < self.polygon_radius)

        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        edge_index = dense_to_sparse(valid)[0]

        rel_vector = transform_point_to_local_coordinate(
            map_pos[edge_index[0]],
            future_flat[edge_index[1]],
            heading_flat[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(map_heading[edge_index[0]] - heading_flat[edge_index[1]])
        heading_valid = map_heading_valid[edge_index[0]]

        edge_features = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
                heading_valid,
            ],
            dim=-1,
        )
        edge_attr = self.map_future_edge_mlp(edge_features)
        return edge_index, edge_attr

    def _build_future_future_edges(
        self,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_mask: torch.Tensor,
        agent_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = future_pos.device
        num_agents, future_steps, _ = future_pos.shape
        if num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        future_flat = future_pos.reshape(-1, 2)
        heading_flat = future_heading.reshape(-1)
        mask_flat = future_mask.reshape(-1)
        future_batch = agent_batch.unsqueeze(1).expand(-1, future_steps).reshape(-1)

        valid = mask_flat.unsqueeze(0) & mask_flat.unsqueeze(1)
        valid = drop_edge_between_samples(valid, batch=future_batch)
        valid = valid & (~torch.eye(valid.size(0), dtype=torch.bool, device=device))

        dist = torch.cdist(future_flat, future_flat)
        valid = valid & (dist < self.agent_radius)

        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        edge_index = dense_to_sparse(valid.contiguous())[0]
        rel_vector = transform_point_to_local_coordinate(
            future_flat[edge_index[0]],
            future_flat[edge_index[1]],
            heading_flat[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(heading_flat[edge_index[0]] - heading_flat[edge_index[1]])
        edge_features = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
            ],
            dim=-1,
        )
        edge_attr = self.future_future_edge_mlp(edge_features)
        return edge_index, edge_attr
