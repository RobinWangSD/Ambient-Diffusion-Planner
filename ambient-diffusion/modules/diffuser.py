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
        num_historical_steps: int = 21,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        temporal_span: int = 6,
        agent_radius: float = 60.0,
        polygon_radius: float = 30.0,
        segment_length: int = 80,
        segment_overlap: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.num_historical_steps = num_historical_steps
        self.temporal_span = temporal_span
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.segment_length = max(1, segment_length) + 1    # considering starting states
        self.segment_overlap = max(0, segment_overlap)  + 1
        if self.segment_overlap >= self.segment_length:
            raise ValueError("segment_overlap must be smaller than segment_length.")

        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.past_future_edge_mlp = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.map_future_edge_mlp = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.future_future_edge_mlp = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.segment_mlp = TwoLayerMLP(
            input_dim=state_dim * self.segment_length,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.segment_pos_embed = SinusoidalTimeEmbedding(hidden_dim)

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
        self.future_agent_attn = nn.ModuleList(
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
        self.future_temporal_attn = nn.ModuleList(
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
        agent_embs: torch.Tensor,
        hist_mask: torch.Tensor,
        x_t: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            data: hetero batch with agent/map attributes.
            map_embeddings: [num_map_nodes, hidden_dim]
            agent_embs: [num_agents, num_historical_steps, hidden_dim]
            hist_mask: bool [num_agents, num_historical_steps]
            x_t: noisy future states [num_agents, num_future_steps, state_dim]
            diffusion_time: [num_agents] or [num_agents, 1]

        Returns:
            Predicted noise with same shape as x_t.
        """
        if x_t.numel() == 0:
            return torch.zeros_like(x_t)

        device = x_t.device
        num_agents, total_steps, _ = x_t.shape

        agent_store = data['agent']
        past_pos = agent_store['history_position']
        past_heading = agent_store['history_heading']

        segments, segment_indices = self._segment_future(x_t, total_steps)
        num_agents, num_segments, _, _ = segments.shape
        seg_position = segments[:, :, 0, :2]
        seg_heading = torch.atan2(segments[:, :, 0, 2], segments[:, :, 0, 3])

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
            torch.zeros(map_embeddings.size(0), device=device, dtype=seg_heading.dtype),
        )
        map_heading_valid = data['polygon'].get(
            'heading_valid_mask',
            torch.ones(map_embeddings.size(0), device=device, dtype=seg_heading.dtype),
        )
        map_pos = data['polygon']['position'][:, :2]

        # assume the model has no knowledge about the dataset quality
        token_mask = torch.ones((segments.shape[:2])).to(dtype=torch.bool, device=device)
        future_time = segment_indices[:, 0].unsqueeze(0).expand(num_agents, -1)

        edges_pf = self._build_past_future_edges(
            past_pos,
            past_heading,
            hist_mask,
            seg_position,
            seg_heading,
            token_mask,
            future_time,
        )
        edges_mf = self._build_map_future_edges(
            map_pos,
            map_heading,
            map_heading_valid,
            map_batch,
            seg_position,
            seg_heading,
            token_mask,
            batch_agents,
        )
        edges_agent, edges_temporal = self._build_future_future_edges(
            seg_position,
            seg_heading,
            token_mask,
            batch_agents,
            future_time,
        )

        segment_emb = self.segment_mlp(segments.view(num_agents, num_segments, -1))
        t_emb = self.time_embed(diffusion_time).unsqueeze(1).expand(-1, num_segments, -1)

        future_flat = segment_emb.reshape(-1, self.hidden_dim)
        past_flat = agent_embs.reshape(-1, self.hidden_dim)

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
            future_flat = self.future_agent_attn[i](
                x=future_flat,
                edge_index=edges_agent[0],
                edge_attr=edges_agent[1],
            )
            future_flat = self.future_temporal_attn[i](
                x=future_flat,
                edge_index=edges_temporal[0],
                edge_attr=edges_temporal[1],
            )

        future_token_emb = future_flat.view(num_agents, num_segments, self.hidden_dim)
        expanded_token_emb = self._detokenize_future(
            future_token_emb,
            segment_indices,
            total_steps,
        )
        pred_noise = self.output_head(expanded_token_emb)
        return pred_noise

    def _segment_future(
        self,
        x_t: torch.Tensor,
        total_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x_t.device

        overlap = self.segment_overlap
        length = self.segment_length
        stride = length - overlap
        num_segments = int((total_steps - overlap) / stride)

        starts = torch.arange(num_segments, device=device, dtype=torch.long) * stride
        indices = starts.unsqueeze(1) + torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
        
        segments = x_t[:, indices]

        return segments, indices

    def _detokenize_future(
        self,
        token_emb: torch.Tensor,
        segment_indices: torch.Tensor,
        segment_valid_mask: torch.Tensor,
        total_steps: int,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = token_emb.shape
        device = token_emb.device
        reconstructed = token_emb.new_zeros(batch_size, total_steps, hidden_dim)
        weights = token_emb.new_zeros(batch_size, total_steps, 1)

        idx_hidden = segment_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        src = token_emb.unsqueeze(2) * segment_valid_mask.unsqueeze(-1).float()
        reconstructed.scatter_add_(1, idx_hidden, src)

        idx_weight = segment_indices.unsqueeze(-1)
        weight_src = segment_valid_mask.unsqueeze(-1).float()
        weights.scatter_add_(1, idx_weight, weight_src)

        reconstructed = reconstructed / weights.clamp(min=1.0)
        return reconstructed

    def _build_past_future_edges(
        self,
        past_pos: torch.Tensor,
        past_heading: torch.Tensor,
        hist_mask: torch.Tensor,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_mask: torch.Tensor,
        future_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = past_pos.device
        num_agents, _, _ = past_pos.shape
        future_steps = future_pos.size(1)

        if num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        hist_idx = torch.arange(num_agents * self.num_historical_steps, device=device).view(num_agents, self.num_historical_steps)
        future_idx = torch.arange(num_agents * future_steps, device=device).view(num_agents, future_steps)
        connectivity = hist_mask.unsqueeze(-1) & future_mask.unsqueeze(1)

        if not connectivity.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device)
            return edge_index, edge_attr

        src = hist_idx.unsqueeze(-1).expand(-1, self.num_historical_steps, future_steps)[connectivity]
        dst = future_idx.unsqueeze(1).expand(-1, self.num_historical_steps, future_steps)[connectivity]

        gather_mask = connectivity
        rel_vector = transform_point_to_local_coordinate(
            past_pos.unsqueeze(-2).expand(-1, -1, future_steps, -1)[gather_mask],
            future_pos.unsqueeze(1).expand(-1, self.num_historical_steps, -1, -1)[gather_mask],
            future_heading.unsqueeze(1).expand(-1, self.num_historical_steps, -1)[gather_mask],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(
            past_heading.unsqueeze(-1).expand(-1, -1, future_steps)[gather_mask]
            - future_heading.unsqueeze(1).expand(-1, self.num_historical_steps, -1)[gather_mask]
        )

        time_hist = torch.arange(self.num_historical_steps, device=device).view(1, self.num_historical_steps, 1).float()
        time_future = future_time.view(num_agents, 1, future_steps)
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

    def build_future_mask(
        self,
        data: Batch,
        num_agents: int,
        total_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        agent_store = data['agent']
        base_mask = agent_store['target_valid_mask'] if 'target_valid_mask' in agent_store else None
        if base_mask is None:
            base_mask = torch.ones(
                (num_agents, max(total_steps - 1, 0)),
                device=device,
                dtype=torch.bool,
            )
        else:
            base_mask = base_mask.to(device=device).bool()
        need_steps = max(total_steps - 1, 0)
        if base_mask.size(1) < need_steps:
            pad = torch.ones(
                (num_agents, need_steps - base_mask.size(1)),
                device=device,
                dtype=base_mask.dtype,
            )
            base_mask = torch.cat([base_mask, pad], dim=1)
        base_mask = base_mask[:, :need_steps]

        if 'current_mask' in agent_store:
            start_mask = agent_store['current_mask'].to(device=device).bool().view(num_agents, 1)
        else:
            start_mask = torch.ones((num_agents, 1), device=device, dtype=torch.bool)
        future_mask = torch.cat([start_mask, base_mask], dim=1)
        if future_mask.size(1) < total_steps:
            pad = torch.ones(
                (num_agents, total_steps - future_mask.size(1)),
                device=device,
                dtype=future_mask.dtype,
            )
            future_mask = torch.cat([future_mask, pad], dim=1)
        else:
            future_mask = future_mask[:, :total_steps]
        return future_mask.bool()

    def _build_future_future_edges(
        self,
        future_pos: torch.Tensor,
        future_heading: torch.Tensor,
        future_mask: torch.Tensor,
        agent_batch: torch.Tensor,
        future_time: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        device = future_pos.device
        num_agents, future_steps, _ = future_pos.shape
        if num_agents == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            empty_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_pos.dtype)
            return (empty_index, empty_attr), (empty_index, empty_attr)

        future_flat = future_pos.reshape(-1, 2)
        heading_flat = future_heading.reshape(-1)
        mask_flat = future_mask.reshape(-1)
        future_batch = agent_batch.unsqueeze(1).expand(-1, future_steps).reshape(-1)
        agent_index = torch.arange(num_agents, device=device).unsqueeze(1).expand(-1, future_steps).reshape(-1)
        time_index = future_time.reshape(-1)

        agent_edges = self._build_agent_agent_edges(
            future_flat,
            heading_flat,
            mask_flat,
            future_batch,
            agent_index,
        )
        temporal_edges = self._build_temporal_edges(
            future_flat,
            heading_flat,
            mask_flat,
            future_batch,
            agent_index,
            time_index,
        )
        return agent_edges, temporal_edges

    def _build_agent_agent_edges(
        self,
        future_flat: torch.Tensor,
        heading_flat: torch.Tensor,
        mask_flat: torch.Tensor,
        future_batch: torch.Tensor,
        agent_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = future_flat.device
        num_nodes = future_flat.size(0)
        if num_nodes == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        valid = mask_flat.unsqueeze(0) & mask_flat.unsqueeze(1)
        valid = drop_edge_between_samples(valid, batch=future_batch)
        same_agent = agent_index.unsqueeze(0) == agent_index.unsqueeze(1)
        valid = valid & (~same_agent)
        valid = valid & (~torch.eye(num_nodes, dtype=torch.bool, device=device))
        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        dist = torch.cdist(future_flat, future_flat)
        valid = valid & (dist < self.agent_radius)
        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        edge_index = dense_to_sparse(valid.contiguous())[0]
        edge_attr = self._compute_future_edge_attr(future_flat, heading_flat, edge_index)
        return edge_index, edge_attr

    def _build_temporal_edges(
        self,
        future_flat: torch.Tensor,
        heading_flat: torch.Tensor,
        mask_flat: torch.Tensor,
        future_batch: torch.Tensor,
        agent_index: torch.Tensor,
        time_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = future_flat.device
        num_nodes = future_flat.size(0)
        if num_nodes == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        valid = mask_flat.unsqueeze(0) & mask_flat.unsqueeze(1)
        valid = drop_edge_between_samples(valid, batch=future_batch)
        same_agent = agent_index.unsqueeze(0) == agent_index.unsqueeze(1)
        valid = valid & same_agent

        delta_t = (time_index.unsqueeze(0) - time_index.unsqueeze(1)).abs()
        valid = valid & (delta_t > 0)
        if self.temporal_span is not None:
            valid = valid & (delta_t <= self.temporal_span)

        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        edge_index = dense_to_sparse(valid.contiguous())[0]
        edge_attr = self._compute_future_edge_attr(future_flat, heading_flat, edge_index)
        return edge_index, edge_attr

    def _compute_future_edge_attr(
        self,
        future_flat: torch.Tensor,
        heading_flat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((0, self.hidden_dim), device=future_flat.device, dtype=future_flat.dtype)

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
        return self.future_future_edge_mlp(edge_features)
