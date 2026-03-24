import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math



def sinusoidal_embedding(k: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional embeddings for diffusion timesteps.

    Args:
        k: Diffusion timestep indices, shape (B,) or (B,1).
           Integer or float tensor representing which denoising iteration we're at.
        d_model: Embedding dimension. Must be even.

    Returns:
        Embedding tensor of shape (B, d_model).
        Even indices use sin, odd indices use cos.
    """
    assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"

    k = k.float().view(-1, 1)  # (B, 1)

    # Dimension indices for each sin/cos pair: [0, 1, 2, ..., d_model/2 - 1]
    i = torch.arange(d_model // 2, device=k.device).float()  # (d_model/2,)

    # Frequencies: 1 / 10000^(2i / d_model)
    # Equivalently: exp(-2i / d_model * log(10000))
    freq = torch.exp(-math.log(10000.0) * (2.0 * i / d_model))  # (d_model/2,)

    # Outer product gives (B, d_model/2) of angles
    angles = k * freq  # (B, d_model/2)

    # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
    emb = torch.stack([angles.sin(), angles.cos()], dim=-1)  # (B, d_model/2, 2)
    emb = emb.view(-1, d_model)  # (B, d_model)

    return emb


class CrossAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout =0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads

        # Q from x, K/V from context
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.scale_factor = math.sqrt(d_model // n_heads)

    def forward(self,x,context):
        """
        x: (B,L_tgt,d_model)
        context: (B,L_src,d_model)

        """

        B, L_tgt, _ = x.shape
        Bc, L_src, _ = context.shape
        assert B == Bc        # "Batch sizes must match"

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(B, L_tgt, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(B, L_src, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(B, L_src, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        # q, k, v: (B, H, L_*, d_head)

        scores = ( q @ k.transpose(-2,-1))/self.scale_factor # (B, H, L_tgt, L_src)

        scores = F.softmax(scores,dim=-1)
        scores = self.attn_dropout(scores)

        y = scores @ v
        y = y.transpose(1,2).contiguous().view(B,L_tgt,self.d_model)
        y = self.out(y)
        return self.residual_dropout(y)







class MHA(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads #4

        self.k_proj = nn.Linear(d_model,d_model)
        self.v_proj = nn.Linear(d_model,d_model)
        self.q_proj = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.scale_factor = math.sqrt(d_model // n_heads)  # Store as parameter


    def forward(self,x):
        batch_size,seq_len,d_model = x.shape


        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size,seq_len,self.n_heads,self.d_model//self.n_heads).transpose(1,2) #(batch_size,n_heads,seq_len,d_model//n_heads)
        q = q.view(batch_size,seq_len,self.n_heads,self.d_model//self.n_heads).transpose(1,2) #(batch_size,n_heads,seq_len,d_model//n_heads)
        v = v.view(batch_size,seq_len,self.n_heads,self.d_model//self.n_heads).transpose(1,2) #(batch_size,n_heads,seq_len,d_model//n_heads)
        
        score = (q @ k.transpose(-2,-1))/self.scale_factor #k.size(-1) = d_model//n_heads = 96, prevents softmax saturation


        score = F.softmax(score,dim=-1) #softmax over the last dimension, acts as gating mechanism
        score = self.attn_dropout(score)
        y = (score @ v) # *B,H,L,L x *B,H,L,D -> *B,H,L,D

        y = y.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model) #(batch_size,seq_len,d_model) get all the heads together side by side

        k = k.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        v = v.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        #output projection

        return self.residual_dropout(self.out(y))


class FeedForward(nn.Module):
    def __init__(self,d_model,dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4,d_model)
        )

    def forward(self,x):

        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        #MHA
        self.mha = MHA(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        #CrossAttention
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        #rest
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,context,cached_k=None,cached_v=None,use_cache=False):
        residual = x

        #MHA
        attn = self.mha(self.norm1(x))

        x = residual + self.dropout1(attn)

        #CrossAttention
        cross_residual = x
        cross_out = self.cross_attn(self.norm_cross(x),context)
        x = cross_residual + self.dropout_cross(cross_out)


        #Rest
        x = x + self.dropout2(self.ffn(self.norm2(x)))

        if use_cache:
            return x,cached_k,cached_v
        else:
            return x


class DiffusionPolicyTransformer(nn.Module):
    def __init__(self, max_seq_len = 16,action_dim = 7,obs_dim = 53, n_heads=4, d_model=128,dropout=0,blocks=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        self.action_embedding = nn.Linear(action_dim,d_model)
        self.obs_embedding = nn.Linear(obs_dim,d_model)


        self.embedding_dropout = nn.Dropout(dropout)
        self.action_pos_embedding = nn.Parameter(torch.randn(max_seq_len + 1, d_model))  # +1 for k token
        self.obs_pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))

        self.blocks = nn.ModuleList([Block(self.d_model,self.n_heads,self.dropout) for _ in range(blocks)])

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model,action_dim) 


    def forward(self, actions, context, diffusion_timesteps):
        x = actions
        seq_len = x.size(1)
        x = self.action_embedding(x) 



        sin_diffstep_emb = sinusoidal_embedding(diffusion_timesteps, self.d_model)
        sin_diffstep_emb = sin_diffstep_emb.unsqueeze(1)

        x = torch.cat([sin_diffstep_emb, x], dim=1) #should be (B, L+1, d_model) now

        x = self.embedding_dropout(x + self.action_pos_embedding[:seq_len + 1])



        context_len = context.size(1)
        context = self.obs_embedding(context)
        context = self.embedding_dropout(context + self.obs_pos_embedding[:context_len]) 



        for block in self.blocks:
            x = block(x,context)


        x = self.layer_norm(x)
        x = self.out(x[:, 1:, :])  # (B, L, action_dim) — skip the k token

        return x