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
    angle_between_2d_vectors,
)


class AgentEncoder(nn.Module):
    """
    Agent history encoder that mirrors QCNet's interaction flow while
    reusing Plan-R1 style GraphAttention + TwoLayerMLP blocks.
    """

    def __init__(
        self,
        num_historical_steps: int = 21,
        hidden_dim: int = 128,
        time_span: int = 10,
        agent_radius: float = 60.0,
        polygon_radius: float = 30.0,
        num_attn_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_historical_steps = num_historical_steps
        self.hidden_dim = hidden_dim
        self.time_span = time_span
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.num_attn_layers = num_attn_layers

        self.state_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.agent_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.identity_embedding = nn.Embedding(2, hidden_dim)

        self.temporal_edge_emb = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.map_edge_emb = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.agent_edge_emb = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.temporal_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.map_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=False,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.agent_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )

    def forward(self, data: Batch, map_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: HeteroData batch containing agent + polygon tensors.
            map_embeddings: Encoded polygon features (from MapEncoder).

        Returns:
            Tuple of (agent_embeddings [Na, T, D], valid_mask [Na, T]).
        """
        agent_store = data['agent']
        
        position = agent_store['history_position']
        heading = agent_store['history_heading']
        velocity = agent_store['history_velocity']
        visible_mask = agent_store['history_mask'].bool()
        
        agent_box = data['agent']['box'] 
        agent_type = data['agent'].get('type', torch.zeros(position.size(0), device=position.device, dtype=torch.long)).long()
        agent_identity = data['agent'].get('identity', torch.zeros(position.size(0), device=position.device, dtype=torch.long)).long()
        batch_agent = data['agent'].get('batch', torch.zeros(position.size(0), device=position.device, dtype=torch.long))

        num_agents = position.shape[0]
        assert num_agents == agent_store['num_nodes']

        head_vector = torch.stack([heading.cos(), heading.sin()], dim=-1)
        motion_vector = torch.cat(
            [position.new_zeros(num_agents, 1, 2), position[:, 1:] - position[:, :-1]],
            dim=1,
        )
        speed_motion = torch.norm(motion_vector, dim=-1)
        angle_motion = angle_between_2d_vectors(head_vector, motion_vector)
        speed_velocity = torch.norm(velocity, dim=-1)
        angle_velocity = angle_between_2d_vectors(head_vector, velocity)

        state_feat = torch.stack([speed_motion, angle_motion, speed_velocity, angle_velocity], dim=-1)
        state_emb = self.state_emb_layer(state_feat.view(-1, state_feat.size(-1))).view(num_agents, self.num_historical_steps, self.hidden_dim)
        static_emb = self.agent_emb_layer(input=agent_box) + self.type_embedding(agent_type) + self.identity_embedding(agent_identity)
        agent_embs = state_emb + static_emb.unsqueeze(1)

        flat_agent_embs = agent_embs.reshape(-1, self.hidden_dim)
        flat_position = position.reshape(-1, 2)
        flat_heading = heading.reshape(-1)
        flat_mask = visible_mask.reshape(-1)
        agent_time_idx = torch.arange(self.num_historical_steps, device=position.device).unsqueeze(0).expand(num_agents, -1).reshape(-1)
        agent_batch_flat = batch_agent.unsqueeze(1).expand(-1, self.num_historical_steps).reshape(-1)

        temporal_edge_index, temporal_edge_attr = self._build_temporal_graph(
            flat_position, flat_heading, visible_mask, agent_time_idx
        )
        map_edge_index, map_edge_attr = self._build_map_graph(
            data, flat_position, flat_heading, agent_batch_flat, flat_mask
        )
        agent_edge_index, agent_edge_attr = self._build_agent_graph(position, heading, visible_mask, batch_agent)

        emb = flat_agent_embs
        for i in range(self.num_attn_layers):
            emb = self.temporal_attn_layers[i](x=emb, edge_index=temporal_edge_index, edge_attr=temporal_edge_attr)
            emb = self.map_attn_layers[i](x=[map_embeddings, emb], edge_index=map_edge_index, edge_attr=map_edge_attr)
            emb = emb.reshape(num_agents, self.num_historical_steps, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            emb = self.agent_attn_layers[i](x=emb, edge_index=agent_edge_index, edge_attr=agent_edge_attr)
            emb = emb.reshape(self.num_historical_steps, num_agents, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

        agent_embs = emb.reshape(num_agents, self.num_historical_steps, self.hidden_dim)
        return agent_embs, visible_mask

    def _build_temporal_graph(
        self,
        flat_position: torch.Tensor,
        flat_heading: torch.Tensor,
        mask: torch.Tensor,
        time_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index = dense_to_sparse(temporal_mask.contiguous())[0]
        if edge_index.numel() == 0:
            edge_attr = flat_position.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr

        time_delta = time_index[edge_index[1]] - time_index[edge_index[0]]
        valid = time_delta > 0
        if self.time_span is not None:
            valid = valid & (time_delta <= self.time_span)
        edge_index = edge_index[:, valid]
        if edge_index.numel() == 0:
            edge_attr = flat_position.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr
        time_delta = time_delta[valid]

        rel_vector = transform_point_to_local_coordinate(
            flat_position[edge_index[0]],
            flat_position[edge_index[1]],
            flat_heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(flat_heading[edge_index[0]] - flat_heading[edge_index[1]])
        attr = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
                time_delta.float(),
            ],
            dim=-1,
        )
        edge_attr = self.temporal_edge_emb(attr)
        return edge_index, edge_attr

    def _build_map_graph(
        self,
        data: Batch,
        flat_position: torch.Tensor,
        flat_heading: torch.Tensor,
        agent_batch_flat: torch.Tensor,
        agent_valid_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_map = data['polygon']['position']
        heading_map = data['polygon']['heading']
        batch_map = data['polygon'].get('batch', torch.zeros(pos_map.size(0), device=pos_map.device, dtype=torch.long))
        heading_valid = data['polygon'].get(
            'heading_valid_mask',
            torch.ones(pos_map.size(0), device=pos_map.device, dtype=torch.float32),
        ).float()

        valid_mask = agent_valid_flat.unsqueeze(0).expand(pos_map.size(0), -1)
        valid_mask = drop_edge_between_samples(valid_mask, batch=(batch_map, agent_batch_flat))
        edge_index = dense_to_sparse(valid_mask.contiguous())[0]
        if edge_index.numel() == 0:
            edge_attr = pos_map.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr

        dist = torch.norm(pos_map[edge_index[0]] - flat_position[edge_index[1]], dim=-1)
        inside_radius = dist < self.polygon_radius
        edge_index = edge_index[:, inside_radius]
        if edge_index.numel() == 0:
            edge_attr = pos_map.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr

        rel_vector = transform_point_to_local_coordinate(
            pos_map[edge_index[0]],
            flat_position[edge_index[1]],
            flat_heading[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(heading_map[edge_index[0]] - flat_heading[edge_index[1]])
        attr = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
                heading_valid[edge_index[0]],
            ],
            dim=-1,
        )
        edge_attr = self.map_edge_emb(attr)
        return edge_index, edge_attr

    def _build_agent_graph(
        self,
        position: torch.Tensor,
        heading: torch.Tensor,
        mask: torch.Tensor,
        batch_agent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_time = position.transpose(0, 1).reshape(-1, 2)
        heading_time = heading.transpose(0, 1).reshape(-1)
        mask_time = mask.transpose(0, 1)

        valid_mask = mask_time.unsqueeze(2) & mask_time.unsqueeze(1)
        valid_mask = drop_edge_between_samples(valid_mask, batch=batch_agent)
        edge_index = dense_to_sparse(valid_mask.contiguous())[0]
        if edge_index.numel() == 0:
            edge_attr = position.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr

        dist = torch.norm(pos_time[edge_index[0]] - pos_time[edge_index[1]], dim=-1)
        inside_radius = dist < self.agent_radius
        edge_index = edge_index[:, inside_radius]
        if edge_index.numel() == 0:
            edge_attr = position.new_zeros((0, self.hidden_dim))
            return edge_index, edge_attr

        rel_vector = transform_point_to_local_coordinate(
            pos_time[edge_index[0]],
            pos_time[edge_index[1]],
            heading_time[edge_index[1]],
        )
        length, theta = compute_angles_lengths_2D(rel_vector)
        heading_delta = wrap_angle(heading_time[edge_index[0]] - heading_time[edge_index[1]])
        attr = torch.stack(
            [
                length,
                torch.cos(theta),
                torch.sin(theta),
                torch.cos(heading_delta),
                torch.sin(heading_delta),
            ],
            dim=-1,
        )
        edge_attr = self.agent_edge_emb(attr)
        return edge_index, edge_attr
