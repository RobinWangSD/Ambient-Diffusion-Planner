import torch
import math
from typing import Tuple


def wrap_angle(angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta


def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix


def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros_like(angle).unsqueeze(-1).repeat_interleave(2,-1).unsqueeze(-1).repeat_interleave(2,-1)
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix


def transform_point_to_local_coordinate(point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point


def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(current_matrix.indices(), torch.ones_like(current_matrix.values()), current_matrix.size())

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)

        current_matrix = next_matrix
    return reach_matrices
