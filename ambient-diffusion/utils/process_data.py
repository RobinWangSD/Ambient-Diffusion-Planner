import torch
import math
from typing import Tuple, Union


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


def drop_edge_between_samples(valid_mask: torch.Tensor,
                              batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Masks out edges that connect nodes from different samples."""
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)

    if valid_mask.ndim == batch_matrix.ndim:
        valid_mask = valid_mask & batch_matrix
    elif valid_mask.ndim == batch_matrix.ndim + 1:
        valid_mask = valid_mask & batch_matrix.unsqueeze(0)
    else:
        raise ValueError("Mismatched shapes between valid_mask and batch assignments.")
    return valid_mask


def angle_between_2d_vectors(ctr_vector: torch.Tensor,
                             nbr_vector: torch.Tensor,
                             eps: float = 1e-6) -> torch.Tensor:
    """Returns signed angle from ctr_vector to nbr_vector."""
    ctr_norm = torch.norm(ctr_vector, dim=-1, keepdim=True).clamp(min=eps)
    nbr_norm = torch.norm(nbr_vector, dim=-1, keepdim=True).clamp(min=eps)
    ctr_unit = ctr_vector / ctr_norm
    nbr_unit = nbr_vector / nbr_norm
    dot = (ctr_unit * nbr_unit).sum(dim=-1).clamp(min=-1.0, max=1.0)
    cross = ctr_unit[..., 0] * nbr_unit[..., 1] - ctr_unit[..., 1] * nbr_unit[..., 0]
    angle = torch.atan2(cross, dot)
    angle = torch.where(torch.norm(nbr_vector, dim=-1) < eps, torch.zeros_like(angle), angle)
    return angle
