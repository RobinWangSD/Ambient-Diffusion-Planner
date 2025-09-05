import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from unitraj.models.base_model.base_model import BaseModel
from .flow_planner_utils import (
    PerceiverEncoder, 
    ChunkEmbedder, 
    TimestepEmbedder, 
    FactorizedDiTBlock,
    FinalLayer,
    RouteEncoder,
)

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FlowPlannerEncoder(nn.Module):
    '''
    Flow Model Ego Centric Encoder using Latent Query Attention.
    '''

    def __init__(self, config):
        super(FlowPlannerEncoder, self).__init__()
        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.k_attr = config['num_agent_feature']
        self.map_attr = config['num_map_feature']
        self.max_num_roads = config['max_num_roads']
        self.d_k = config['hidden_size']
        self.num_queries_enc = config['num_queries_enc']
        self._M = config['num_observed_agents']  # num agents without the ego-agent
        self.past_T = config['past_len']
        self.num_heads = config['tx_num_heads']
        self.num_encoder_layers = config['num_encoder_layers']

        # Ego-centric representation 
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.perceiver_encoder = PerceiverEncoder(
            self.num_queries_enc, self.d_k,
            num_cross_attention_heads=self.num_heads,
            num_cross_attention_qk_channels=self.d_k,
            num_cross_attention_v_channels=self.d_k,
            num_cross_attention_layers=self.num_encoder_layers,
            num_self_attention_qk_channels=self.d_k,
            num_self_attention_v_channels=self.d_k,
            num_self_attention_blocks=self.num_encoder_layers,

            )

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )

        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))

        self.selu = nn.SELU(inplace=True)

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        # env_masks = (1.0 - env_masks_orig).to(torch.bool)
        # env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
        #                                                                            -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.  (B, t_h, n_oa+1)
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states (B, T_h, n_oa, 29)

        return ego_tensor, opps_tensor, opps_masks

    def scene_encoding(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            encoding: shape [B, N_enc_q, h] 
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1  
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)     # (B, T_h, N_oa+1, 29)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))              # (B, T_h, N_ca+1, 29)
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k) # (B, T_h * N_ca+1, h)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,  # (B, N_r*N_pts, h)
                                                                                                            self.d_k))# + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)
        return context

    def forward(self, batch):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        
        # Obtain Ego agent state and mask. The following slicing only work for ego centric setting
        batch_size, num_observed_agents, t_hist, in_feat_dim = agents_in.shape
        ego_in = torch.gather(
            agents_in, 
            1,
            torch.zeros(
                batch_size, 
                dtype=torch.int64, 
                device=agents_in.device,
                ).view((-1, 1, 1, 1)).repeat(1, 1, t_hist, in_feat_dim)
        ).squeeze(1)
        ego_mask = torch.gather(
            agents_mask, 
            1,
            torch.zeros(
                batch_size, 
                dtype=torch.int64,
                device=agents_in.device,
                ).view((-1, 1, 1)).repeat(1, 1, t_hist)
        ).squeeze(1)

        # slice num_observe_agents 
        agents_in = agents_in[:, :self._M, :, :]
        agents_mask = agents_mask[:, :self._M, :]

        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)   # (B, N_observed_agents, T_h, 30)
        agents_in = agents_in.transpose(1, 2)                                   # (B, T_h, N_oa, 30)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)            # (B, T_h, 30)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads

        encodings = self.scene_encoding(model_input)    # (B, N_enc_query, h) ego centric encoding of the scene
        
        return encodings


class FlowPlannerDecoder(nn.Module):
    """
    Flow model decoder that takes in ego centric scene embdedings from the encoder.
    """
    def __init__(self, config):
        super(FlowPlannerDecoder, self).__init__()
        self.config = config
        self.d_k = config['dec_hidden_size']
        self.future_T = config['future_len']
        self.chunk_size = config['chunk_size']
        self.action_len = config['action_len']
        self.action_type = config['action_type']
        self.depth = config['dit_depth']
        self.mlp_ratio = config['mlp_ratio']
        self.num_heads = config['num_heads']

        self.num_modeled_agents = config['num_modeled_agents']
        if self.action_type == 'trajectory':
            self.feature_size = 3       # (x, y, yaw)
        elif self.action_type == "velocity":
            self.feature_size = 3       # (vx, vy, yaw_rate)
        elif self.action_type == "ik":
            self.feature_size = 2       # (acceleration, yaw_rate)


        assert self.future_T % self.action_len == 0
        assert self.future_T // self.action_len % self.chunk_size == 0
        self.chunk_steps = self.future_T // self.action_len // self.chunk_size
        self.agent_chunk_encoder = ChunkEmbedder(
            timesteps = self.future_T // self.action_len,
            chunk_size = self.chunk_size,
            feature_size = self.feature_size,
            hidden_dim = self.d_k,
        )

        self.agent_embedding = nn.Embedding(2, self.d_k)
        self.agent_pos_embedding = nn.Sequential(
            nn.Linear(in_features=7, out_features=self.d_k),
            nn.SELU(),
            )

        self.t_embedder = TimestepEmbedder(self.d_k)

        self.blocks = nn.ModuleList(
            [
                FactorizedDiTBlock(
                    dim = self.d_k, 
                    heads = self.num_heads,
                    mlp_ratio = self.mlp_ratio,
                    ) for i in range(self.depth)
            ]
        )

        self.final_layer = FinalLayer(self.d_k, self.chunk_size * self.feature_size)

        self.route_encoder = RouteEncoder(
            config['max_num_route_lanes'], 
            config['max_points_per_lane'], 
            drop_path_rate=config['route_encoder_drop_path_rate'], 
            hidden_dim=self.d_k,
            )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize ChunkEmbedder
        w = self.agent_chunk_encoder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.agent_chunk_encoder.proj.bias, 0)

        # Initialize agent embedding 
        nn.init.normal_(self.agent_embedding.weight, std=0.02)

        # Zero-out agent_pos_embedding Linear
        nn.init.constant_(self.agent_pos_embedding[0].weight, 0)
        nn.init.constant_(self.agent_pos_embedding[0].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation1[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation1[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation2[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.proj[1].weight, 0)
        nn.init.constant_(self.final_layer.proj[1].bias, 0)
        nn.init.constant_(self.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.final_layer.proj[-1].bias, 0)

    def unpatchify(self, x):
        x = x[:, :, 1:, :]      # first T represents curr states.
        B, A, T_chunk, _ = x.shape

        x = x.reshape(shape=(B, A, T_chunk, self.chunk_size, self.feature_size))
        actions = x.reshape(shape=(B, A, T_chunk * self.chunk_size, self.feature_size))
        return actions

    def forward(self, x, t, encodings, curr_states, agent_mask, y_goal=None, y_route=None):
        """
        Forward pass of decoder. 
        Args: 
            x: noised actions (B, A, T, h)
            t: flow timesteps (B, A)
            encodings: scene encodings (B, N_enc_query, h+3)
            curr_states: current states of modeled agents (B, A, 7)
            agent_mask: boolean mask denoting whether a modeled agent is valid at current time (B, A)
            y: route guidance TODO
        """
        # projecting noisy trajectories
        B, A, T, _ = x.shape
        x = x.permute(0, 3, 1, 2)   #(B, s, A, T)   
        chunked_x = self.agent_chunk_encoder(x) # (B, h, A, T//chunk_size)
        x = chunked_x.permute(0, 2, 3, 1).view(B, A, self.chunk_steps, self.d_k)   # (B, A, T//chunk_size, h)
        
        curr_states_embedding = self.agent_pos_embedding(curr_states)   # (B, A, h)
        curr_states_embedding = curr_states_embedding[:, :, None, :]   # (B, A, 1, h)
        x = torch.cat((curr_states_embedding, x), dim=2) # (B, A, T_chunk+1, h)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :], 
                self.agent_embedding.weight[1][None, :].expand(A - 1, -1)
            ], dim=0
            )  # (A, h)
        x_embedding = x_embedding[None, :, None, :].expand(B, -1, self.chunk_steps+1, -1) # (B, A, T_chunk+1, h)
        x = x + x_embedding     # (B, A, T_chunk+1, h)

        t_embedding = self.t_embedder(t)        # (B, h)
        # TODO: add route condition and goal point condition
        y = t_embedding     #(B, h)
        if y_route is not None:
            route_embedding = self.route_encoder(y_route)
            y = y + route_embedding

        for block in self.blocks:
            x = block(x, encodings, y, agent_mask)

        x = self.final_layer(x, y)
        x = self.unpatchify(x)
        return x
        

        








        

        
