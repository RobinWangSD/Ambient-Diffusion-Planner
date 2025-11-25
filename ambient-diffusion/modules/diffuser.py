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
    generate_counterclockwise_rotation_matrix,
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

    model_type = "x_start"  # used by dpm-solver wrapper

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
        normalize_segments: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_future_steps = num_future_steps
        self.num_historical_steps = num_historical_steps
        self.temporal_span = temporal_span
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.segment_length = max(1, segment_length)   # considering starting states
        self.segment_overlap = max(0, segment_overlap)
        self.normalize_segments = normalize_segments
        if self.segment_overlap >= self.segment_length:
            raise ValueError("segment_overlap must be smaller than segment_length.")

        self.num_heads = num_heads
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
        self.time_condition_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

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

        self.output_head = nn.Linear(hidden_dim, self.segment_length * self.state_dim)

    def forward(
        self,
        data: Batch,
        map_embeddings: torch.Tensor,
        agent_embs: torch.Tensor,
        hist_mask: torch.Tensor,
        x_t: torch.Tensor,
        current_states: torch.Tensor,
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
        stride = self.segment_length - self.segment_overlap

        num_agents, num_segments, seg_len, _ = x_t.shape
        starts = torch.arange(num_segments, device=device, dtype=torch.long) * stride
        segment_indices = starts.unsqueeze(1) + torch.arange(seg_len, device=device, dtype=torch.long)
        total_steps = stride * (num_segments - 1) + seg_len
        assert total_steps == self.num_future_steps
        segments = x_t

        agent_store = data['agent']
        past_pos = agent_store['history_position']
        past_heading = agent_store['history_heading']

        segment_valid_mask = torch.ones((num_agents, num_segments, seg_len), dtype=torch.bool, device=device)

        if self.normalize_segments:
            starting_states = segments.new_empty((num_agents, num_segments, 4))
            global_segments = torch.empty_like(segments)
            current_state = current_states
            for idx in range(num_segments):
                heading = torch.atan2(current_state[:, 3], current_state[:, 2]).unsqueeze(1)
                rotation = generate_counterclockwise_rotation_matrix(heading)

                local_seg = segments[:, idx]
                pos = torch.matmul(rotation, local_seg[..., :2].unsqueeze(-1)).squeeze(-1) + current_state[:, None, :2]
                cos = local_seg[..., 2] * current_state[:, None, 2] - local_seg[..., 3] * current_state[:, None, 3]
                sin = local_seg[..., 3] * current_state[:, None, 2] + local_seg[..., 2] * current_state[:, None, 3]

                global_segments[:, idx] = torch.stack([pos[..., 0], pos[..., 1], cos, sin], dim=-1)
                starting_states[:, idx] = current_state

                if idx + 1 < num_segments:
                    current_state = global_segments[:, idx, stride - 1]
        else:
            heading = torch.atan2(current_states[:, 3], current_states[:, 2]).view(num_agents, 1, 1)
            rotation = generate_counterclockwise_rotation_matrix(heading)
            pos = torch.matmul(rotation, segments[..., :2].unsqueeze(-1)).squeeze(-1) + current_states[:, None, None, :2]
            cos = segments[..., 2] * current_states[:, None, None, 2] - segments[..., 3] * current_states[:, None, None, 3]
            sin = segments[..., 3] * current_states[:, None, None, 2] + segments[..., 2] * current_states[:, None, None, 3]
            global_segments = torch.stack([pos[..., 0], pos[..., 1], cos, sin], dim=-1)

            starting_states = segments.new_empty((num_agents, num_segments, 4))
            starting_states[:, 0] = current_states
            if num_segments > 1:
                starting_states[:, 1:] = global_segments[:, :-1, stride - 1]

        heading_dtype = starting_states.dtype

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
            torch.zeros(map_embeddings.size(0), device=device, dtype=heading_dtype),
        )
        map_heading_valid = data['polygon'].get(
            'heading_valid_mask',
            torch.ones(map_embeddings.size(0), device=device, dtype=heading_dtype),
        )
        map_pos = data['polygon']['position'][:, :2]

        # build edges for segment tokens anchored at their starting states
        token_mask = segment_valid_mask[..., 0]
        future_time_hist = starts.view(1, -1).expand(num_agents, -1)
        future_time_seg = torch.arange(num_segments, device=device, dtype=torch.long).unsqueeze(0).expand(num_agents, -1)
        edges_pf = self._build_past_future_edges(
            past_pos,
            past_heading,
            hist_mask,
            starting_states,
            token_mask,
            future_time_hist,
        )
        edges_mf = self._build_map_future_edges(
            map_pos,
            map_heading,
            map_heading_valid,
            map_batch,
            starting_states,
            token_mask,
            batch_agents,
        )
        edges_agent, edges_temporal = self._build_future_future_edges(
            starting_states,
            token_mask,
            batch_agents,
            future_time_seg,
        )

        segment_emb = self.segment_mlp(segments.view(num_agents, num_segments, -1))
        t_emb = self.time_embed(diffusion_time).unsqueeze(1)
        segment_emb = segment_emb + self.time_condition_attn(
            query=segment_emb,
            key=t_emb,
            value=t_emb,
            need_weights=False,
        )[0]

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
        return self._detokenize_future(
            future_token_emb,
            segment_indices,
            segment_valid_mask,
            total_steps,
        )

    def _detokenize_future(
        self,
        token_emb: torch.Tensor,
        segment_indices: torch.Tensor,
        segment_valid_mask: torch.Tensor,
        total_steps: int,
    ) -> torch.Tensor:
        batch_size, num_segments, _ = token_emb.shape
        pred = self.output_head(token_emb)
        pred = pred.view(batch_size, num_segments, self.segment_length, self.state_dim)
        return pred * segment_valid_mask.unsqueeze(-1).float()

    def _build_past_future_edges(
        self,
        past_pos: torch.Tensor,
        past_heading: torch.Tensor,
        hist_mask: torch.Tensor,
        future_state: torch.Tensor,
        future_mask: torch.Tensor,
        future_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = past_pos.device
        num_agents, hist_steps, _ = past_pos.shape
        future_steps = future_state.size(1)

        if num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_state.dtype)
            return edge_index, edge_attr

        hist_idx = torch.arange(num_agents * hist_steps, device=device).view(num_agents, hist_steps)
        future_idx = torch.arange(num_agents * future_steps, device=device).view(num_agents, future_steps)
        connectivity = hist_mask.unsqueeze(-1) & future_mask.unsqueeze(1)

        if not connectivity.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_state.dtype)
            return edge_index, edge_attr

        agent_ids, hist_ids, future_ids = connectivity.nonzero(as_tuple=True)
        src = hist_idx[agent_ids, hist_ids]
        dst = future_idx[agent_ids, future_ids]

        future_pos = future_state[..., :2]
        future_heading = torch.atan2(future_state[..., 3], future_state[..., 2])
        rel_vector = transform_point_to_local_coordinate(
            past_pos[agent_ids, hist_ids],
            future_pos[agent_ids, future_ids],
            future_heading[agent_ids, future_ids],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(
            past_heading[agent_ids, hist_ids] - future_heading[agent_ids, future_ids]
        )

        time_hist = torch.arange(hist_steps, device=device, dtype=length.dtype)
        delta_t = future_time[agent_ids, future_ids].to(length.dtype) - time_hist[hist_ids]

        edge_features = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
                delta_t,
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
        future_state: torch.Tensor,
        future_mask: torch.Tensor,
        agent_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = map_pos.device
        dtype = future_state.dtype
        num_map = map_pos.size(0)
        num_agents, future_steps, _ = future_state.shape
        if num_map == 0 or num_agents == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            return edge_index, edge_attr

        future_pos = future_state[..., :2]
        future_heading = torch.atan2(future_state[..., 3], future_state[..., 2])

        future_flat = future_pos.reshape(-1, 2)
        heading_flat = future_heading.reshape(-1)
        mask_flat = future_mask.reshape(-1)
        if not mask_flat.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            return edge_index, edge_attr

        future_batch = agent_batch.unsqueeze(1).expand(-1, future_steps).reshape(-1)
        valid_future = mask_flat.nonzero(as_tuple=False).squeeze(1)
        future_flat = future_flat[valid_future]
        heading_flat = heading_flat[valid_future]
        future_batch = future_batch[valid_future]

        if future_flat.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            return edge_index, edge_attr

        batch_match = map_batch.unsqueeze(1) == future_batch.unsqueeze(0)
        if not batch_match.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            return edge_index, edge_attr

        # dist = torch.cdist(map_pos, future_flat)
        # valid = (dist < self.polygon_radius) & batch_match
        valid = batch_match

        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=dtype)
            return edge_index, edge_attr

        map_idx, future_local_idx = valid.nonzero(as_tuple=True)
        future_idx = valid_future[future_local_idx]

        rel_vector = transform_point_to_local_coordinate(
            map_pos[map_idx],
            future_flat[future_local_idx],
            heading_flat[future_local_idx],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(map_heading[map_idx] - heading_flat[future_local_idx])
        heading_valid = map_heading_valid[map_idx]

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
        edge_index = torch.stack([map_idx, future_idx], dim=0)
        return edge_index, edge_attr

    def _build_future_future_edges(
        self,
        future_state: torch.Tensor,
        future_mask: torch.Tensor,
        agent_batch: torch.Tensor,
        future_time: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        device = future_state.device
        num_agents, future_steps, _ = future_state.shape
        if num_agents == 0:
            empty_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            empty_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_state.dtype)
            return (empty_index, empty_attr), (empty_index, empty_attr)

        future_pos = future_state[..., :2]
        future_heading = torch.atan2(future_state[..., 3], future_state[..., 2])
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
        
        if not valid.any():
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, self.hidden_dim), device=device, dtype=future_flat.dtype)
            return edge_index, edge_attr

        # dist = torch.cdist(future_flat, future_flat)
        # valid = valid & (dist < self.agent_radius)
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
