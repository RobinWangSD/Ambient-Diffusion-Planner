import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import (
    TimestepEmbedder, 
    FactorizedDiTBlock, 
    DiTBlock_, 
    DiTBlock, 
    FinalLayer, 
    FinalLayerFactorized, 
    FinalLayerFactorizedV2
    )


class FactorizedDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        if config.action_type == 'traj':
            self.feature_size = 4       # (x, y, sin, cos)
        elif config.action_type == "vel":
            self.feature_size = 4       # (vx, vy, sin_yaw_rate, cos)
        elif config.action_type == "ik":
            self.feature_size = 3       # (acceleration, sin_yaw_rate, cos)
        if config.if_factorized:
            output_dim = self.feature_size
        else:
            output_dim = self.feature_size * (self._future_len + 1)
        self.dit = FactorizedDiT(
            sde=self._sde, 
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= output_dim,
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type,
            chunk_size=config.chunk_size, 
            chunk_overlap=config.chunk_overlap,
            future_len=self._future_len, 
            action_len=config.action_len, 
            action_type=config.action_type,
            decoder_agent_attn_mask=config.decoder_agent_attn_mask,
            use_chunking=config.use_chunking,
            if_factorized=config.if_factorized,
            use_causal_attn=config.use_causal_attn,
            use_agent_validity_in_temporal=config.use_agent_validity_in_temporal,
            use_chunk_t_embed=config.use_chunk_t_embed,
            ego_separate=config.ego_separate,
            key_padding=config.key_padding,
            pad_left=config.pad_left,
            pad_history=config.pad_history,
            v2=config.v2,
            residual_emb=config.residual_emb,
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
        self._guidance_fn = getattr(config, 'guidance_fn', None)
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training or 'sampled_trajectories' in inputs:
            sampled_trajectories = inputs['sampled_trajectories'] # [B, 1 + predicted_neighbor_num, (V_future), 4]
            diffusion_time = inputs['diffusion_time']
            neighbor_future_mask = inputs["neighbor_future_mask"]

            return {
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        current_states,
                        ego_neighbor_encoding,
                        route_lanes,
                        neighbor_current_mask,
                        neighbor_future_mask,
                    ).reshape(B, P, -1, 4)
                }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            temp = inputs['temperature']
            xT = torch.randn(B, P, self._future_len, 4).to(current_states.device) * temp

            def initial_state_constraint(xt, t, step):
                return xt.reshape(B, P, -1, 4)
            
            neighbor_future_mask = None
            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        diffusion_steps = inputs['diffusion_steps'],
                        other_model_params={
                            "current_states": current_states,
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                            "neighbor_future_mask": neighbor_future_mask,                        
                        },
                        dpm_solver_params={
                            "correcting_xt_fn":initial_state_constraint,
                        },
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": ego_neighbor_encoding, 
                                    "route_lanes": route_lanes,
                                    "neighbor_current_mask": neighbor_current_mask,
                                    "neighbor_future_mask": neighbor_future_mask,                           
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))

            return {
                    "prediction": x0
                }


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde, 
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth, 
            output_dim= (config.future_len + 1) * 4, # x, y, cos, sin
            hidden_dim=config.hidden_dim, 
            heads=config.num_heads, 
            dropout=dpr,
            model_type=config.diffusion_model_type
        )
        
        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        
        self._guidance_fn = getattr(config, 'guidance_fn', None)
        
    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1) # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            diffusion_time = inputs['diffusion_time']

            return {
                    "score": self.dit(
                        sampled_trajectories, 
                        diffusion_time,
                        ego_neighbor_encoding,
                        route_lanes,
                        neighbor_current_mask
                    ).reshape(B, P, -1, 4)
                }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = torch.cat([current_states[:, :, None], torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5], dim=2).reshape(B, P, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)
            
            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding, 
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask                            
                        },
                        dpm_solver_params={
                            "correcting_xt_fn":initial_state_constraint,
                        },
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": ego_neighbor_encoding, 
                                    "route_lanes": route_lanes,
                                    "neighbor_current_mask": neighbor_current_mask                            
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer
                            },
                            "guidance_scale": 0.5,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            return {
                    "prediction": x0
                }

        
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class ChunkEmbedder(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            timesteps,
            chunk_size=1,
            chunk_overlap=1,
            feature_size=4,
            hidden_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(feature_size, hidden_dim, kernel_size=(1,chunk_size+chunk_overlap), stride=(1,chunk_size), bias=bias)
        self.norm = norm_layer(hidden_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)    # (B, hidden_dim, A, T)
        x = self.norm(x)
        return x


class FactorizedDiT(nn.Module):
    def __init__(
        self, 
        sde: SDE, 
        route_encoder: nn.Module, 
        depth, 
        output_dim, 
        hidden_dim=192, 
        heads=6, 
        dropout=0.1, 
        mlp_ratio=4.0, 
        model_type="x_start", 
        chunk_size=1, 
        chunk_overlap=1,
        future_len=80, 
        action_len=1, 
        action_type='traj',
        decoder_agent_attn_mask=False,
        use_chunking=False,
        if_factorized=False,
        use_causal_attn=False, 
        use_agent_validity_in_temporal=False,
        use_chunk_t_embed=False,
        ego_separate=False,
        key_padding=False,
        pad_left=False,
        pad_history=False,
        v2=False,
        residual_emb=False,
        ):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        # abort the following and include it in chunking
        # self.preproj_curr_states = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.use_chunking = use_chunking

        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.use_chunk_t_embed = use_chunk_t_embed
        self.ego_separate = ego_separate

        self.if_factorized = if_factorized

        self.pad_history = pad_history
        self.v2 = v2
        self.residual_emb = residual_emb
        if if_factorized:
            if use_chunking:
                self.chunk_len = future_len // action_len // self.chunk_size
                self.padding_size = self.chunk_len * self.chunk_size + self.chunk_overlap - 1 - future_len
                if pad_left:
                    self.zero_padding = nn.ZeroPad2d((self.padding_size, 0, 0, 0))
                else:
                    self.zero_padding = nn.ZeroPad2d((0, self.padding_size, 0, 0))
                if self.use_chunk_t_embed:
                    self.chunk_t_embedding = nn.Embedding(self.chunk_len, hidden_dim)
                if self.ego_separate:
                    self.preproj_futu_states_ego = ChunkEmbedder(
                        timesteps = future_len // action_len,
                        chunk_size = self.chunk_size,
                        chunk_overlap = self.chunk_overlap,
                        feature_size = output_dim,
                        hidden_dim = hidden_dim,
                    )

                self.preproj_futu_states = ChunkEmbedder(
                    timesteps = future_len // action_len,
                    chunk_size = self.chunk_size,
                    chunk_overlap = self.chunk_overlap,
                    feature_size = output_dim,
                    hidden_dim = hidden_dim,
                )
                if self.v2:
                    self.final_layer = FinalLayerFactorizedV2(hidden_dim * self.chunk_len, hidden_dim, output_dim * self.chunk_size * self.chunk_len)
                    if self.ego_separate:
                        self.final_layer_ego = FinalLayerFactorizedV2(hidden_dim * self.chunk_len, hidden_dim, output_dim * self.chunk_size * self.chunk_len)

                else:
                    if self.ego_separate:
                        self.final_layer_ego = FinalLayerFactorized(hidden_dim, output_dim * self.chunk_size)
                    self.final_layer = FinalLayerFactorized(hidden_dim, output_dim * self.chunk_size)
            else:
                self.chunk_size = 1
                self.chunk_len = future_len // action_len // self.chunk_size
                self.preproj_futu_states = self.preproj_curr_states
                self.final_layer = FinalLayerFactorized(hidden_dim, output_dim)
            self.blocks = nn.ModuleList(
                [
                    FactorizedDiTBlock(
                        dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout, 
                        mlp_ratio=mlp_ratio,
                        decoder_agent_attn_mask=decoder_agent_attn_mask,
                        use_causal_attn=use_causal_attn, 
                        use_agent_validity_in_temporal=use_agent_validity_in_temporal,
                        key_padding=key_padding,
                        ) for i in range(depth)
                ]
            )
        else:
            self.chunk_size = 1
            self.chunk_len = future_len // action_len // self.chunk_size
            self.preproj_curr_states = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
            self.preproj_futu_states = self.preproj_curr_states
            self.final_layer = FinalLayer(hidden_dim, output_dim)
            self.blocks = nn.ModuleList(
                [
                    DiTBlock_(
                        dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout, 
                        mlp_ratio=mlp_ratio,
                        ) for i in range(depth)
                ]
            )
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        self.output_dim = output_dim
        self.future_len = future_len
               
    @property
    def model_type(self):
        return self._model_type

    def unpatchify(self, x):
        B, A, T_chunk, _ = x.shape
        assert T_chunk == self.chunk_len

        x = x.reshape(shape=(B, A, T_chunk, self.chunk_size, self.output_dim))
        actions = x.reshape(shape=(B, A, T_chunk * self.chunk_size, self.output_dim))
        return actions
    
    def unpatchify_v2(self, x):
        B, A, _ = x.shape
        actions = x.reshape(shape=(B, A, self.chunk_len * self.chunk_size, self.output_dim))
        return actions

    def forward(self, x, t, current_states, cross_c, route_lanes, neighbor_current_mask, neighbor_future_mask):
        """
        Forward pass of DiT.
        x: (B, P, T, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        x_curr = current_states[:, :, None, :]  # (B, P, 1, 4)
        B, P, T, _ = x.shape

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :], 
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1)
            ], dim=0
            )  # (P, D)
        if self.if_factorized:
            # x_curr = self.preproj_curr_states(x_curr)   # (B, P, 1, D)

            # if self.use_chunking:
            #     x_futu = x.permute(0, 3, 1, 2).contiguous()   #(B, 4, P, T_f) 
            #     if self.ego_separate:
            #         x_futu_ego = self.preproj_futu_states_ego(x_futu[:, :, :1, :])
            #         x_futu_other = self.preproj_futu_states(x_futu[:, :, 1:, :])
            #         x_futu = torch.cat([x_futu_ego, x_futu_other], dim=2)
            #     else:
            #         x_futu = self.preproj_futu_states(x_futu)   # (B, P, T_f, D)
            #     x_futu = x_futu.permute(0, 2, 3, 1).contiguous()       # (B, P, T_chunk, D)
            # else:
            #     x_futu = self.preproj_futu_states(x)

            # x = torch.cat([x_curr, x_futu], dim=2)  # (B, P, T_chunk+1, D)
            # B, P, T, _ = x.shape
            # x_embedding = x_embedding[None, :, None, :].expand(B, -1, self.chunk_len + 1, -1) # (B, P, T_chunk+1, D)
            x = torch.cat([x_curr, x], dim=2)   # (B, P, T_f+1, 4)
            if self.use_chunking:
                # padding at the end
                x = x.permute(0, 3, 1, 2).contiguous()   #(B, 4, P, T_f+1) 
                x = self.zero_padding(x)
                if self.pad_history:
                    x[:, :, :, :self.padding_size] = x[:, :, :, self.padding_size:self.padding_size+1].repeat_interleave(self.padding_size, dim=3)
                if self.ego_separate:
                    x_ego = self.preproj_futu_states_ego(x[:, :, :1, :])
                    x_other = self.preproj_futu_states(x[:, :, 1:, :])
                    x = torch.cat([x_ego, x_other], dim=2)
                else:
                    # print(x.shape)
                    x = self.preproj_futu_states(x)   #(B, 4, P, T_chunk)
                    # print(x.shape)
                x = x.permute(0, 2, 3, 1).contiguous()       # (B, P, T_chunk, D)
            else:
                x = self.preproj_futu_states(x)

            B, P, T, _ = x.shape
            x_embedding = x_embedding[None, :, None, :].expand(B, -1, self.chunk_len, -1) # (B, P, T_chunk+1, D)
        else:
            x = torch.cat([x_curr, x], dim=2).view(B, P, -1)
            x = self.preproj_curr_states(x)
            x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)

        if self.use_chunking and self.use_chunk_t_embed:
            chunk_t_embedding = self.chunk_t_embedding.weight[None, None, :, :].expand(B, P, -1, -1) # (B, P, T_chunk+1, D)
            x = x + chunk_t_embedding
        x = x + x_embedding     

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)      

        agent_curr_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        agent_curr_mask[:, 1:] = neighbor_current_mask

        if self.if_factorized and neighbor_future_mask is not None:
            agent_futu_mask = torch.zeros((B, P, T), dtype=torch.bool, device=x.device)
            neighbor_future_mask = neighbor_future_mask.view(B, P - 1, -1, self.chunk_size)
            assert neighbor_future_mask.shape[2] == T
            agent_futu_mask[:, 1:] = neighbor_future_mask.all(dim=-1)
        else:
            agent_futu_mask = None
        
        for block in self.blocks:
            x = block(x, cross_c, y, agent_curr_mask, agent_futu_mask)  
            if self.residual_emb and self.use_chunking and self.use_chunk_t_embed:
                x = x + chunk_t_embedding
        
        if self.ego_separate:
            # print(x[:, :, :1, :].shape)
            x_ego = self.final_layer_ego(x[:, :1, :, :], y)
            x_other = self.final_layer(x[:, 1:, :, :], y)
            x = torch.cat([x_ego, x_other], dim=1)
        else:
            x = self.final_layer(x, y)
        if self.if_factorized:
            if self.use_chunking:
                if self.v2:
                    x = self.unpatchify_v2(x)      
                else: 
                    x = self.unpatchify(x)      
            else:
                x = x
        else:
            x = x.view(B, P, self.future_len + 1, -1)[:, :, 1:, :]
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape
        
        x = self.preproj(x)

        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        x = x + x_embedding     

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)      

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)  
            
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
