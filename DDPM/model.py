import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    args = timesteps[:, None].to(torch.float32) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], axis=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels

        self.seq1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(num_groups=8, num_channels=out_channels),
            torch.nn.SiLU()
        )

        self.seq2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(num_groups=8, num_channels=out_channels),
            torch.nn.SiLU() 
        )


        self.temb_linear = torch.nn.Linear(temb_channels, out_channels)
        self.channel_proj = torch.nn.Conv2d(in_channels, out_channels, 1) 



    def forward(self, x, temb):
        h = self.seq1(x)

        temb = self.temb_linear(temb)
        h += temb[:, :, None, None] #h is BxDxHxW, temb is BxDx1x1

        h = self.seq2(h)

        if self.in_channels != self.out_channels:
            x = self.channel_proj(x)
        return x + h
    
class Downsample(torch.nn.Module):

    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
 
        return self.conv(x)

class Upsample(torch.nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2) 
        x = self.conv(x)
        return x
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.blocks_per_dim = blocks_per_dim

        temb_channels = hidden_dims[0] * 4
        
        self.emb_seq = nn.Sequential(
            nn.Linear(hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )


        self.first_Conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)


        self.down_residual_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        prev_ch = hidden_dims[0]
        down_block_chans = [prev_ch]
        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(self.blocks_per_dim):
                self.down_residual_blocks.append(
                    ResidualBlock(prev_ch, hidden_dim, temb_channels)
                )
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)
            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_ch))
                prev_ch = hidden_dim
                down_block_chans.append(prev_ch)

        self.bottom_block1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], temb_channels)
        self.bottom_block2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], temb_channels)


        self.up_residual_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()


        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(blocks_per_dim + 1):
                dch = down_block_chans.pop() 
                self.up_residual_blocks.append(
                    ResidualBlock(prev_ch + dch, hidden_dim, temb_channels)
                )
                prev_ch = hidden_dim
                if i and j == blocks_per_dim:
                    self.up_blocks.append(Upsample(prev_ch))
                    prev_ch = hidden_dim

        self.final_seq = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, in_channels, 3, padding=1)
        )


    def forward(self, x, t):
        #Given x, t
    
        emb = timestep_embedding(t, self.hidden_dims[0]) 
        emb = self.emb_seq(emb)

        h = self.first_Conv(x)
        hs = [h]
       
        block_idx = 0
        down_block_idx = 0
        for i in range(len(self.hidden_dims)):
            for _ in range(self.blocks_per_dim):
                h = self.down_residual_blocks[block_idx](h, emb)
                block_idx += 1
                hs.append(h)

            if i != len(self.hidden_dims) - 1:
                h = self.down_blocks[down_block_idx](h)
                down_block_idx += 1
                hs.append(h)

        h = self.bottom_block1(h, emb)
        h = self.bottom_block2(h, emb)

        block_idx = 0
        up_block_idx = 0
        for i in range(len(self.hidden_dims) - 1, -1, -1): #back up through the hidden dimensions
            for j in range(self.blocks_per_dim + 1):
                h = self.up_residual_blocks[block_idx](torch.cat([h, hs.pop()], dim=1), emb)
                block_idx += 1

                if i and j == self.blocks_per_dim:
                    h = self.up_blocks[up_block_idx](h)
                    up_block_idx += 1

        h = self.final_seq(h)
        return h
                    

    
    
