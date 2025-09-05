import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, attn_mask):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x


class DiTBlock_(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, agent_curr_mask, agent_future_mask):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=agent_curr_mask)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x


def modulate_factorized(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :, :1], x[:, :, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

    return x

class FactorizedDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(
        self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, 
        decoder_agent_attn_mask=False, 
        use_causal_attn=False, 
        use_agent_validity_in_temporal=False, 
        key_padding=False
        ):
        super().__init__()
        self.decoder_agent_future_attn_mask = decoder_agent_attn_mask
        self.use_causal_attn = use_causal_attn
        self.use_agent_validity_in_temporal = use_agent_validity_in_temporal
        self.key_padding = key_padding
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # agent attention with AdaptiveLN Zero
        self.norm3 = nn.LayerNorm(dim)
        self.agent_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # history attention block
        self.norm5 = nn.LayerNorm(dim)
        self.history_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm6 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        # Pre-compute and register causal mask as buffer
        self.max_seq_len = 100  # Adjust based on your max sequence length
        self.register_buffer('causal_mask', 
                           torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), 
                                    diagonal=1).bool())


    def forward(self, x, cross_c, y, agent_curr_mask, agent_future_mask):
        B, A, T, h = x.shape

        # causal
        all_modulations = torch.chunk(
            torch.cat([
                self.adaLN_modulation1(y),
                self.adaLN_modulation2(y),
                self.adaLN_modulation3(y)
            ], dim=1), 
            18, dim=1
        )
        
        # Unpack modulations
        shift_msa1, scale_msa1, gate_msa1, shift_mlp1, scale_mlp1, gate_mlp1 = all_modulations[:6]
        shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = all_modulations[6:12]
        shift_msa3, scale_msa3, gate_msa3, shift_mlp3, scale_mlp3, gate_mlp3 = all_modulations[12:18]

        causal_mask = self.causal_mask[:T, :T]
        modulated_x = modulate_factorized(self.norm1(x), shift_msa1, scale_msa1).reshape(B*A, T, h)
        
        if self.use_agent_validity_in_temporal and not self.key_padding:
            agent_future_mask_ = agent_future_mask.reshape(B*A, T)
            temporal_valid_mask = torch.logical_or(
                agent_future_mask_.unsqueeze(1), agent_future_mask_.unsqueeze(2)
                ) # (B*A, T, T)

        if self.use_causal_attn:
            if self.key_padding:
                if self.use_agent_validity_in_temporal:
                    temporal_valid_mask = agent_future_mask.reshape(B*A, T)
                    x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x, attn_mask=causal_mask, key_padding_mask=temporal_valid_mask)[0]
                else:
                    x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x, attn_mask=causal_mask)[0]
            else:
                if self.use_agent_validity_in_temporal:
                    causal_mask = causal_mask.unsqueeze(0).expand(B*A, -1, -1)
                    causal_mask = torch.logical_or(temporal_valid_mask, causal_mask)
                    causal_mask = causal_mask.repeat_interleave(self.heads, dim=0).float() * -1e9
                x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x, attn_mask=causal_mask)[0]
        else:
            if self.use_agent_validity_in_temporal:
                if self.key_padding:
                    temporal_valid_mask = agent_future_mask.reshape(B*A, T)
                    x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x, key_padding_mask=temporal_valid_mask)[0]
                else:
                    temporal_valid_mask = temporal_valid_mask.repeat_interleave(self.heads, dim=0).float() * -1e9
                    x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x, attn_mask=temporal_valid_mask)[0]
            else:
                x_attn = self.temporal_attn(modulated_x, modulated_x, modulated_x)[0]
            
        x = x + gate_msa1.view(B, 1, 1, h) * x_attn.reshape(B, A, T, h)

        modulated_x = modulate_factorized(self.norm2(x), shift_mlp1, scale_mlp1)
        x = x + gate_mlp1.view(B, 1, 1, h) * self.mlp1(modulated_x)

        # agent
        x = x.transpose(1, 2)       # (B, T, A, h)
        modulated_x = modulate_factorized(self.norm3(x), shift_msa2, scale_msa2).reshape(B*T, A, h)

        agent_curr_mask = agent_curr_mask.unsqueeze(1).expand(-1, T, -1).to(x.device) # (B, T, A)
        agent_curr_mask = agent_curr_mask.reshape(B*T, A)
        
        if self.decoder_agent_future_attn_mask: 
            # print('using decoder agent future mask')
            if agent_future_mask is not None:
                agent_future_mask_ = agent_future_mask.transpose(1, 2).reshape(B*T, A)
                agent_attn_mask = torch.logical_or(agent_future_mask_, agent_curr_mask)
            else:
                agent_attn_mask = agent_curr_mask
            if self.key_padding:
                x_attn = self.agent_attn(modulated_x, modulated_x, modulated_x, key_padding_mask=agent_attn_mask)[0]
            else:
                agent_attn_mask = torch.logical_or(
                    agent_attn_mask.unsqueeze(1), agent_attn_mask.unsqueeze(2)
                    ) # (B*T, A, A)
                agent_attn_mask = agent_attn_mask.repeat_interleave(self.heads, dim=0)
                additive_mask = agent_attn_mask.float() * -1e9
                x_attn = self.agent_attn(modulated_x, modulated_x, modulated_x, attn_mask=additive_mask)[0]
        else:
            x_attn = self.agent_attn(modulated_x, modulated_x, modulated_x, key_padding_mask=agent_curr_mask)[0]
        
        x = x + gate_msa2.view(B, 1, 1, h) * x_attn.reshape(B, T, A, h) 

        modulated_x = modulate_factorized(self.norm4(x), shift_mlp2, scale_mlp2)
        x = x + gate_mlp2.view(B, 1, 1, h) * self.mlp2(modulated_x)
        x = x.transpose(1, 2)       # (B, A, T, h)

        # historical
        modulated_x = modulate_factorized(self.norm5(x), shift_msa3, scale_msa3).reshape(B, A*T, h)
        x_attn = self.history_attn(modulated_x, cross_c, cross_c)[0]
        x = x + gate_msa3.view(B, 1, 1, h) * x_attn.reshape(B, A, T, h)
        modulated_x = modulate_factorized(self.norm6(x), shift_mlp3, scale_mlp3)
        x = x + gate_mlp3.view(B, 1, 1, h) * self.mlp3(modulated_x)

        x = x.view(B, A, T, h)
        return x
    
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class FinalLayerFactorized(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, int(hidden_size / 4), bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(int(hidden_size / 4)),
            nn.Linear(int(hidden_size / 4), output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, T, _ = x.shape
        
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate_factorized(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class FinalLayerFactorizedV2(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, y_hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(y_hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, int(hidden_size / 4), bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(int(hidden_size / 4)),
            nn.Linear(int(hidden_size / 4), output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(y_hidden_size, 2 * y_hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, T, _ = x.shape
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate_factorized(self.norm_final(x), shift, scale)
        x = x.view(B, P, -1)
        x = self.proj(x)
        return x