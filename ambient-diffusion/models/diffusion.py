import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from typing import Tuple
import math
from copy import deepcopy
import os

from layers import TwoLayerMLP
from modules import (
    MapEncoder,
    AgentEncoder,
    Diffuser,
)

from diffusion_utils import VPSDE_linear, dpm_sampler
from utils import transform_point_to_local_coordinate

class DiffusionPredictor(pl.LightningModule):
    def __init__(self,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_hops: int = 4,
                 dropout: float = 0.1,
                 agent_time_span: int = 10,
                 agent_radius: float = 60.0,
                 agent_polygon_radius: float = 30.0,
                 agent_num_attn_layers: int = 3,
                 agent_num_heads: int = 8,
                 agent_dropout: float = 0.1,
                 diffuser_num_layers: int = 3,
                 diffuser_num_heads: int = 8,
                 diffuser_dropout: float = 0.1,
                 diffuser_temporal_span: int = 6,
                 diffuser_agent_radius: float = 60.0,
                 diffuser_polygon_radius: float = 30.0,
                 diffuser_segment_length: int = 80,
                 diffuser_segment_overlap: int = 0,
                 diffuser_normalize_segments: bool = True,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0001,
                 warmup_epochs: int = 4) -> None:
        super(DiffusionPredictor, self).__init__()
        self.save_hyperparameters()

        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hops = num_hops
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.segment_length = diffuser_segment_length
        self.segment_overlap = diffuser_segment_overlap
        self.normalize_segments = diffuser_normalize_segments

        # Map encoder from PlanR1
        self.map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.agent_encoder = AgentEncoder(
            num_historical_steps=num_historical_steps,
            hidden_dim=hidden_dim,
            time_span=agent_time_span,
            agent_radius=agent_radius,
            polygon_radius=agent_polygon_radius,
            num_attn_layers=agent_num_attn_layers,
            num_heads=agent_num_heads,
            dropout=agent_dropout,
        )
        self.denoiser = Diffuser(
            state_dim=4,
            hidden_dim=hidden_dim,
            num_future_steps=num_future_steps,
            num_historical_steps=num_historical_steps,
            num_layers=diffuser_num_layers,
            num_heads=diffuser_num_heads,
            dropout=diffuser_dropout,
            temporal_span=diffuser_temporal_span,
            agent_radius=diffuser_agent_radius,
            polygon_radius=diffuser_polygon_radius,
            segment_length=diffuser_segment_length,
            segment_overlap=diffuser_segment_overlap,
            normalize_segments=diffuser_normalize_segments,
        )

        # diffusion utils
        self._sde = VPSDE_linear()

    @property
    def sde(self):
        return self._sde

    def training_step(self, data: Batch, batch_idx: int) -> None:
         
        target = data['agent']['target']        # ([N1, N2, ...], T_f, 4)
        target_mask = data['agent']['target_valid_mask']
        t = data['agent']['diffusion_time']     # ([N1, N2, ...])

        # segment trajectories
        segments, segments_mask, indices = self.create_segments(target, target_mask, self.num_future_steps)  # ([N1, N2, ...], n_seg, T_seg, 4)
        A, S, T, D = segments.shape
        segments_mask = segments_mask.bool()

        if self.normalize_segments:
            segments_target = segments.clone()
            # transform each segment to its local coordinate frame
            starting_states = segments.new_zeros((A, S, 1, D))
            starting_states[:, 0, 0, :] = data['agent']['current_states']
            if S > 1:
                prev_indices = indices[1:, 0] - 1
                prev_states = target.index_select(1, prev_indices)
                starting_states[:, 1:, 0, :] = prev_states

            if indices.shape[0] != S or indices.shape[1] != self.segment_length:
                raise ValueError("Segment indices shape does not match the configured segment layout.")
            if indices[-1, -1].item() != self.num_future_steps - 1:
                raise ValueError("Segment indices do not cover the full prediction horizon.")

            start_pos = starting_states[..., :2]
            start_heading = torch.atan2(starting_states[..., 3], starting_states[..., 2])
            local_pos = transform_point_to_local_coordinate(
                segments[..., :2],
                start_pos,
                start_heading,
            )

            start_cos = starting_states[..., 2]
            start_sin = starting_states[..., 3]
            rel_cos = segments[..., 2] * start_cos + segments[..., 3] * start_sin
            rel_sin = segments[..., 3] * start_cos - segments[..., 2] * start_sin

            transformed_segments = torch.cat(
                [local_pos, rel_cos.unsqueeze(-1), rel_sin.unsqueeze(-1)],
                dim=-1,
            )
            segments = torch.where(
                segments_mask.unsqueeze(-1),
                transformed_segments,
                segments,
            )
        else:
            # transform trajectories to the coordinate of current states
            start_pos = data['agent']['current_states'][..., :2].unsqueeze(1).unsqueeze(1)
            start_heading = torch.atan2(
                data['agent']['current_states'][..., 3],
                data['agent']['current_states'][..., 2],
            ).unsqueeze(1).unsqueeze(1)

            local_pos = transform_point_to_local_coordinate(
                segments[..., :2],
                start_pos,
                start_heading,
            )
            start_cos = data['agent']['current_states'][..., 2].unsqueeze(1).unsqueeze(1)
            start_sin = data['agent']['current_states'][..., 3].unsqueeze(1).unsqueeze(1)
            rel_cos = segments[..., 2] * start_cos + segments[..., 3] * start_sin
            rel_sin = segments[..., 3] * start_cos - segments[..., 2] * start_sin

            transformed_segments = torch.cat(
                [local_pos, rel_cos.unsqueeze(-1), rel_sin.unsqueeze(-1)],
                dim=-1,
            )
            segments = torch.where(
                segments_mask.unsqueeze(-1),
                transformed_segments,
                segments,
            )

        # add noise 
        z = torch.randn_like(segments, device=target.device)  # ([N1, N2, ...], n_seg, T_seg, 4)
        mean, std = self._sde.marginal_prob(segments, t)  
        x_t = mean + std * z

        # Encode map features using PlanR1's MapEncoder
        map_embeddings = self.map_encoder(data)  # Returns [(M1,...,Mb), hidden_dim]
        agent_embs, visible_mask = self.agent_encoder(data, map_embeddings)

        pred_clean = self.denoiser(
            data=data,
            map_embeddings=map_embeddings,
            agent_embs=agent_embs,
            hist_mask=visible_mask,
            x_t=x_t,
            current_states=data['agent']['current_states'],
            diffusion_time=t,
        )

        weight = segments_mask.float()
        if self.normalize_segments:
            stride = self.segment_length - self.segment_overlap
            global_segments = []
            state_start = data['agent']['current_states']
            
            for idx in range(S):
                local = pred_clean[:, idx]
                cos_h = state_start[:, 2].unsqueeze(1)
                sin_h = state_start[:, 3].unsqueeze(1)
                pos_x = local[..., 0] * cos_h - local[..., 1] * sin_h + state_start[:, None, 0]
                pos_y = local[..., 0] * sin_h + local[..., 1] * cos_h + state_start[:, None, 1]
                cos_theta = local[..., 2] * cos_h - local[..., 3] * sin_h
                sin_theta = local[..., 3] * cos_h + local[..., 2] * sin_h
                
                global_seg = torch.stack([pos_x, pos_y, cos_theta, sin_theta], dim=-1)
                global_segments.append(global_seg)
                
                if idx + 1 < S:
                    state_start = global_seg[:, stride - 1]
            
            global_pred = torch.stack(global_segments, dim=1)
            mse = (global_pred - segments_target) ** 2
        else:
            mse = (pred_clean - segments) ** 2

        denom = weight.sum().clamp(min=1.0)
        recon_loss = (mse * weight.unsqueeze(-1)).sum() / denom

        overlap = self.segment_overlap
        if overlap > 0 and pred_clean.size(1) > 1:
            slice1 = pred_clean[:, :-1, -overlap:, :]
            slice2 = pred_clean[:, 1:, :overlap, :]
            mask1 = segments_mask[:, :-1, -overlap:]
            mask2 = segments_mask[:, 1:, :overlap]
            valid = (mask1 & mask2).float().unsqueeze(-1)

            consistency_diff = (slice1 - slice2) ** 2 * valid
            consistency_denom = valid.sum().clamp(min=1.0)
            consistency_loss = consistency_diff.sum() / consistency_denom
        else:
            consistency_loss = segments.new_zeros(())

        loss = recon_loss + consistency_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.size(0) if target.size(0) else 1)

        return loss

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        target = data['agent']['target']
        target_mask = data['agent']['target_valid_mask']

        segments, segments_mask, indices = self.create_segments(target, target_mask, self.num_future_steps)
        A, S, L, D = segments.shape
        if A == 0 or S == 0:
            return

        map_embeddings = self.map_encoder(data)
        agent_embs, visible_mask = self.agent_encoder(data, map_embeddings)
        current_states = data['agent']['current_states']

        class _DenoiserWrapper(nn.Module):
            def __init__(self, denoiser, data, map_embeddings, agent_embs, hist_mask, current_states):
                super().__init__()
                self.denoiser = denoiser
                self.model_type = denoiser.model_type
                self.data = data
                self.map_embeddings = map_embeddings
                self.agent_embs = agent_embs
                self.hist_mask = hist_mask
                self.current_states = current_states

            def forward(self, x, t):
                return self.denoiser(
                    data=self.data,
                    map_embeddings=self.map_embeddings,
                    agent_embs=self.agent_embs,
                    hist_mask=self.hist_mask,
                    x_t=x,
                    current_states=self.current_states,
                    diffusion_time=t,
                )

        denoiser_wrapper = _DenoiserWrapper(
            self.denoiser,
            data,
            map_embeddings,
            agent_embs,
            visible_mask,
            current_states,
        )

        with torch.no_grad():
            x_T = torch.randn_like(segments)
            sampled_segments = dpm_sampler(
                denoiser_wrapper,
                x_T,
                diffusion_steps=10,
            )

        stride = self.segment_length - self.segment_overlap
        if self.normalize_segments:
            global_segments = torch.empty_like(sampled_segments)
            state_start = current_states
            for idx in range(S):
                local = sampled_segments[:, idx]
                cos_h = state_start[:, 2].unsqueeze(1)
                sin_h = state_start[:, 3].unsqueeze(1)
                pos_x = local[..., 0] * cos_h - local[..., 1] * sin_h + state_start[:, None, 0]
                pos_y = local[..., 0] * sin_h + local[..., 1] * cos_h + state_start[:, None, 1]
                cos_theta = local[..., 2] * cos_h - local[..., 3] * sin_h
                sin_theta = local[..., 3] * cos_h + local[..., 2] * sin_h
                global_segments[:, idx] = torch.stack([pos_x, pos_y, cos_theta, sin_theta], dim=-1)
                if idx + 1 < S:
                    state_start = global_segments[:, idx, stride - 1]
        else:
            cos_h = current_states[:, 2].view(-1, 1, 1)
            sin_h = current_states[:, 3].view(-1, 1, 1)
            pos_x = sampled_segments[..., 0] * cos_h - sampled_segments[..., 1] * sin_h + current_states[:, None, None, 0]
            pos_y = sampled_segments[..., 0] * sin_h + sampled_segments[..., 1] * cos_h + current_states[:, None, None, 1]
            cos_theta = sampled_segments[..., 2] * cos_h - sampled_segments[..., 3] * sin_h
            sin_theta = sampled_segments[..., 3] * cos_h + sampled_segments[..., 2] * sin_h
            global_segments = torch.stack([pos_x, pos_y, cos_theta, sin_theta], dim=-1)

        pred = torch.zeros(A, self.num_future_steps, D, device=segments.device, dtype=segments.dtype)
        counts = torch.zeros(A, self.num_future_steps, 1, device=segments.device, dtype=segments.dtype)

        idx_flat = indices.reshape(1, -1).expand(A, -1)
        mask_flat = segments_mask.reshape(A, -1).unsqueeze(-1).float()
        seg_flat = global_segments.reshape(A, -1, D)
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, D)

        pred.scatter_add_(1, idx_expanded, seg_flat * mask_flat)
        counts.scatter_add_(1, idx_flat.unsqueeze(-1), mask_flat)
        counts = counts.clamp(min=1.0)
        pred = pred / counts

        pred_pos = pred[..., :2]
        target_pos = target[..., :2]
        target_mask_float = target_mask.float()

        dist = torch.norm(pred_pos - target_pos, dim=-1)
        ade_agent = (dist * target_mask_float).sum(dim=-1) / target_mask_float.sum(dim=-1).clamp(min=1.0)
        valid_agents = target_mask.any(dim=-1)
        min_ade = ade_agent[valid_agents].mean() if valid_agents.any() else torch.zeros((), device=dist.device, dtype=dist.dtype)

        fde = dist[:, -1] * target_mask_float[:, -1]
        denom_fde = target_mask_float[:, -1].sum().clamp(min=1.0)
        min_fde = fde.sum() / denom_fde

        self.log('val_minADE', min_ade, prog_bar=True, on_step=False, on_epoch=True, batch_size=A)
        self.log('val_minFDE', min_fde, prog_bar=True, on_step=False, on_epoch=True, batch_size=A)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['sde'] = {
            'beta_min': getattr(self._sde, '_beta_min', 0.1),
            'beta_max': getattr(self._sde, '_beta_max', 20.0),
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        sde_cfg = checkpoint.get('sde')
        if sde_cfg is None:
            self._sde = VPSDE_linear()
            return

        beta_min = sde_cfg.get('beta_min', 0.1)
        beta_max = sde_cfg.get('beta_max', 20.0)
        self._sde = VPSDE_linear(beta_max=beta_max, beta_min=beta_min)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters():
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(decay)], "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = self.warmup_epochs
        T_max = getattr(self.trainer, "max_epochs", warmup_epochs)

        def warmup_cosine_annealing_schedule(epoch: int):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    def create_segments(
        self,
        x_t: torch.Tensor,
        x_t_valid_mask: torch.Tensor,
        total_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x_t.device

        overlap = self.segment_overlap
        length = self.segment_length
        stride = length - overlap
        assert (total_steps - overlap) % stride == 0
        num_segments = int((total_steps - overlap) / stride)

        starts = torch.arange(num_segments, device=device, dtype=torch.long) * stride
        indices = starts.unsqueeze(1) + torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
        
        segments = x_t[:, indices]
        segments_valid_mask = x_t_valid_mask[:, indices]

        return segments, segments_valid_mask, indices#
