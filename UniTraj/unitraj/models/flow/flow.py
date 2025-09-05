import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR

from unitraj.models.base_model.base_model import BaseModel
from .flow_ego_centric_modules import EgoCentricFlowEncoder, EgoCentricFlowDecoder
from .path import CondOTProbPath
from .solver import ODESolver, ModelWrapper


class EgoCentricFlow(BaseModel):
    '''
    Ego centric Flow Model.
    '''

    def __init__(self, config):
        super(EgoCentricFlow, self).__init__(config)
        self.config = config
        self.action_type = config['action_type']
        self.future_len = config['future_len']
        self.action_len = config['action_len']
        self.action_type = config['action_type']
        if self.action_type == 'trajectory':
            self.feature_size = 3       # (x, y, yaw)
        elif self.action_type == "velocity":
            self.feature_size = 3       # (vx, vy, yaw_rate)
        elif self.action_type == "ik":
            self.feature_size = 2       # (acceleration, yaw_rate)
        self.num_samples = config['num_samples']
        self.num_steps = config['num_steps']

        assert self.action_type in ['trajectory']   # currently only supports trajectory type
        
        self.encoder = EgoCentricFlowEncoder(self.config)
        self.decoder = EgoCentricFlowDecoder(self.config)

        self.path = CondOTProbPath() # use probability path defined in rectified flow 

        self.model_wrapper = ModelWrapper(self.decoder)
        self.solver = ODESolver(self.model_wrapper) # TODO: 

    def cosine_annealing_warm_up_restarts(self, optimizer, epoch, warm_up_epoch, start_factor=0.1):
        assert epoch >= warm_up_epoch
        T_warmup = warm_up_epoch
        
        warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
        fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
        
        scheduler = SequentialLR(optimizer, 
                                schedulers=[warmup_scheduler, fixed_scheduler], 
                                milestones=[T_warmup])
        return scheduler

    def configure_optimizers(self):
        """
        Initialize optimizer and learning rate scheduler. This function is handled by lightning.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        scheduler = self.cosine_annealing_warm_up_restarts(
            optimizer = optimizer, 
            epoch = self.config['max_epochs'], 
            warm_up_epoch = self.config['warm_up_epoch'],
            )
        return [optimizer], [scheduler]

    def forward(self, x_t, t, batch, curr_states, agent_mask):
        """
        Single forward pass of the flow model FM(x_t, t, c), used for training only. 
        Sample with flow model should directly call sample() method. 

        Args:
            x_t: noised agent trajectories (B, N_max_agents, T_f+1, A)
            t: flow time step (B, A)
            batch: dict
        """
        encodings = self.encoder(batch)     # (B, N_env_query, h)

        # decoder takes scene encodings and output velocity
        v_t = self.decoder(x_t, t, encodings, curr_states, agent_mask)
        return v_t

    def get_loss(self, batch):
        batch_size = batch['batch_size']

        inputs = batch['input_dict']
        # track_index_to_predict = inputs['track_index_to_predict']
        obj_trajs = inputs['obj_trajs']      # (B, A, T_h, s)
        B, A, T_h, feat_size = obj_trajs.shape
        track_index_to_predict = inputs['track_index_to_predict']   # (B, A)
        index_valid_mask = track_index_to_predict == -1     # position with True indicates value is zero (B, A)
        track_index_to_predict[index_valid_mask] = 0    # torch gather does not support -1 as index
        _track_index_to_predict = track_index_to_predict.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_h, feat_size).to(torch.int64) # (B, A, T_h, s)
        obj_trajs = torch.gather(obj_trajs, dim=1, index=_track_index_to_predict)  
        obj_trajs[index_valid_mask] = 0.
        curr_states = obj_trajs[:, :, -1, [0, 1, 23, 24, 6, 7, 8]] # (B, A, (x, y, sin, cos, type_one_hot))

        obj_trajs_mask = inputs['obj_trajs_mask'] # (B, A, T)
        _track_index_to_predict = track_index_to_predict.unsqueeze(-1).expand(-1, -1, T_h).to(torch.int64) # (B, A, T_h)
        obj_trajs_mask = torch.gather(obj_trajs_mask, dim=1, index=_track_index_to_predict)
        obj_trajs_mask[index_valid_mask] = False
        agent_mask = obj_trajs_mask[:, :, -1]   # (B, A)

        # TODO: add other types
        # if self.action_type == "trajectory":
        #     gt = batch['input_dict']['center_gt_trajs']
        gt_trajs = inputs['center_gt_trajs'][:, :, :, :3]       # (B, A, T_f, (x, y, yaw))
        gt_trajs_mask = inputs['center_gt_trajs_mask']          # (B, A, T_f)
        x_0 = torch.randn(gt_trajs.shape, device=gt_trajs.device)
        t = torch.rand(batch_size, device=gt_trajs.device)
               
        sample = self.path.sample(t=t, x_0=x_0, x_1=gt_trajs)
        x_t = sample.x_t
        dx_t = sample.dx_t

        v = self.forward(x_t, t, batch, curr_states, agent_mask==0)

        # compute loss with conditional flow matching
        state_loss = F.smooth_l1_loss(v[..., :2], dx_t[..., :2], reduction='none').sum(-1)
        yaw_error = v[..., 2] - dx_t[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)
        
        state_loss = state_loss * gt_trajs_mask
        yaw_loss = yaw_loss * gt_trajs_mask

        # Calculate the mean loss
        state_loss_mean = state_loss.sum() / gt_trajs_mask.sum()
        yaw_loss_mean = yaw_loss.sum() / gt_trajs_mask.sum()
        return {
            f'yaw_loss': yaw_loss_mean,
            f'state_loss': state_loss_mean,
            f'fm_loss': yaw_loss_mean + state_loss_mean,
        }

    def training_step(self, batch, batch_idx):
        """
        Training step of the flow model.
        1. sample time t with batch size b
        2. construct targets with probability path
        3. compute flow matching loss
        # TODO: support more action type training
        """
        loss_dict = self.get_loss(batch)
        for k, v in loss_dict.items():
            self.log("train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

        return loss_dict['fm_loss']

    def validation_step(self, batch, batch_idx):
        """
        Validating random denoising steps.
        """
        loss_dict = self.get_loss(batch)
        for k, v in loss_dict.items():
            self.log("val/" + k, v, on_step=False, on_epoch=True, sync_dist=True)
        return loss_dict['fm_loss']

    def test_step(self, batch, batch_idx):
        # this is not the exact ade, fde, miss rate over the entire dataset. this is averaged over scenarios.
        sample, agent_mask = self.sample(batch, num_samples=self.num_samples, num_steps=self.num_steps) # (B, num_samples, A, T, feat_size)
        
        inputs = batch['input_dict']
        gt_traj = inputs['center_gt_trajs'].unsqueeze(1)  # .transpose(0, 1).unsqueeze(0)
        gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
        center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']

        predicted_traj = sample

        # Calculate ADE losses
        ade_diff = torch.norm(predicted_traj[:, :, :, :, :2] - gt_traj[:, :, :, :, :2], 2, dim=-1)
        ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / (torch.sum(gt_traj_mask, dim=-1)+1e-9)
        ade_losses = ade_losses.cpu().detach().numpy()
        agent_mask = agent_mask.cpu().detach().numpy().reshape(-1)
        minade = np.min(ade_losses, axis=1).reshape(-1)[agent_mask]
        # Calculate FDE losses
        bs, modes, num_agent, future_len = ade_diff.shape
        center_gt_final_valid_idx = center_gt_final_valid_idx.view(bs, 1, num_agent, 1).repeat(1, modes, 1, 1).to(torch.int64)

        fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
        minfde = np.min(fde, axis=1).reshape(-1)[agent_mask]

        miss_rate = (minfde > 2.0)

        loss_dict = {
            'minADE6': minade,
            'minFDE6': minfde,
            'miss_rate': miss_rate.astype(np.float32),
            }

        size_dict = {key: len(value) for key, value in loss_dict.items()}
        loss_dict = {key: np.mean(value) for key, value in loss_dict.items()}
        for k, v in loss_dict.items():
            self.log("test/" + k, v, on_step=False, on_epoch=True, sync_dist=True, batch_size=size_dict[k])

    def sample(self, batch, num_samples=1, num_steps=10):
        """
        Full sampling process from source (Gaustion) distribution.
        """
        batch_size = batch['batch_size']

        inputs = batch['input_dict']
        # track_index_to_predict = inputs['track_index_to_predict']
        obj_trajs = inputs['obj_trajs']      # (B, A, T_h, s)   A should be equal to max_agents
        B, A, T_h, feat_size = obj_trajs.shape
        track_index_to_predict = inputs['track_index_to_predict']   # (B, A)
        index_valid_mask = track_index_to_predict == -1     # position with True indicates value is zero (B, A)
        track_index_to_predict[index_valid_mask] = 0    # torch gather does not support -1 as index
        _track_index_to_predict = track_index_to_predict.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_h, feat_size).to(torch.int64) # (B, A, T_h, s)
        obj_trajs = torch.gather(obj_trajs, dim=1, index=_track_index_to_predict)  
        obj_trajs[index_valid_mask] = 0.
        curr_states = obj_trajs[:, :, -1, [0, 1, 23, 24, 6, 7, 8]] # (B, A, (x, y, sin, cos, type_one_hot))

        obj_trajs_mask = inputs['obj_trajs_mask'] # (B, A, T)
        _track_index_to_predict = track_index_to_predict.unsqueeze(-1).expand(-1, -1, T_h).to(torch.int64) # (B, A, T_h)
        obj_trajs_mask = torch.gather(obj_trajs_mask, dim=1, index=_track_index_to_predict)
        obj_trajs_mask[index_valid_mask] = False
        agent_mask = obj_trajs_mask[:, :, -1]   # (B, A)

        assert self.future_len % self.action_len == 0
        T_f = self.future_len // self.action_len
        
        samples = []
        for i in range(num_samples):
            encodings = self.encoder(batch)

            x_0 = torch.randn((batch_size, A, T_f, self.feature_size)).to(encodings.device)

            x_1 = self.solver.sample(
                x_init=x_0, 
                method='euler', 
                step_size=1/num_steps, 
                encodings=encodings,
                curr_states=curr_states,
                agent_mask=agent_mask==0,
                )   # (B, A, T, feat_size)
            
            samples.append(x_1[:, :, :, :])
        
        samples = torch.stack(samples, axis=1) # (B, num_samples, A, T, feat_size)
        return samples, agent_mask







