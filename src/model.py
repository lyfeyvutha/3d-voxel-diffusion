import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math

# --- Time Embedding ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# --- Simple Building Block ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.dropout = nn.Identity()
        
        if in_ch == out_ch:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, t):
        residual = self.res_conv(x)
        
        h = F.silu(self.bn1(self.conv1(x)))
        
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 3]
        h = h + time_emb
        
        h = F.silu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        
        return h + residual

# --- Simple U-Net ---
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # --- Encoder ---
        self.down1 = Block(in_channels, 32, time_emb_dim)
        self.pool1 = nn.MaxPool3d(2)
        
        self.down2 = Block(32, 64, time_emb_dim)
        self.pool2 = nn.MaxPool3d(2)
        
        self.down3 = Block(64, 128, time_emb_dim)
        self.pool3 = nn.MaxPool3d(2)
        
        # --- Bottleneck ---
        self.bot1 = Block(128, 256, time_emb_dim)
        self.bot2 = Block(256, 256, time_emb_dim)

        # --- Decoder ---
        # Level 1: 256 -> 128
        self.upconv1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up1 = Block(128 + 128, 128, time_emb_dim)  # Concatenated: 256 total
        
        # Level 2: 128 -> 64
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up2 = Block(64 + 64, 64, time_emb_dim)     # Concatenated: 128 total
        
        # Level 3: 64 -> 32
        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up3 = Block(32 + 32, 32, time_emb_dim)     # Concatenated: 64 total
        
        # --- Final Output ---
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Process time embedding
        t = self.time_mlp(t)
        
        # Encoder path
        x1 = self.down1(x, t)     # (B, 32, 64, 64, 64)
        x2 = self.pool1(x1)       # (B, 32, 32, 32, 32)
        
        x3 = self.down2(x2, t)    # (B, 64, 32, 32, 32)
        x4 = self.pool2(x3)       # (B, 64, 16, 16, 16)
        
        x5 = self.down3(x4, t)    # (B, 128, 16, 16, 16)
        x6 = self.pool3(x5)       # (B, 128, 8, 8, 8)

        # Bottleneck
        x = self.bot1(x6, t)      # (B, 256, 8, 8, 8)
        x = self.bot2(x, t)       # (B, 256, 8, 8, 8)

        # Decoder path with skip connections
        x = self.upconv1(x)           # (B, 128, 16, 16, 16)
        x = torch.cat([x, x5], dim=1) # (B, 256, 16, 16, 16)
        x = self.up1(x, t)            # (B, 128, 16, 16, 16)
        
        x = self.upconv2(x)           # (B, 64, 32, 32, 32)
        x = torch.cat([x, x3], dim=1) # (B, 128, 32, 32, 32)
        x = self.up2(x, t)            # (B, 64, 32, 32, 32)
        
        x = self.upconv3(x)           # (B, 32, 64, 64, 64)
        x = torch.cat([x, x1], dim=1) # (B, 64, 64, 64, 64)
        x = self.up3(x, t)            # (B, 32, 64, 64, 64)
        
        return self.out(x)            # (B, 1, 64, 64, 64)
