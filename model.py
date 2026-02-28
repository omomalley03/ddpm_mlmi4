"""
U-Net architecture for DDPM (Ho et al. 2020).

Based on the PixelCNN++ / Wide ResNet backbone with:
- Pre-activation residual blocks (GroupNorm → SiLU → Conv)
- Sinusoidal timestep embeddings
- Single-head self-attention at 16×16 resolution
- Skip connections via concatenation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Transformer-style sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Pre-activation residual block with timestep conditioning.

    GroupNorm → SiLU → Conv → (+time_emb) → GroupNorm → SiLU → Dropout → Conv + skip
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head self-attention with GroupNorm and residual."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        h_flat = h_norm.view(b, c, h * w)

        q = self.q(h_flat)
        k = self.k(h_flat)
        v = self.v(h_flat)

        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj_out(out)
        out = out.view(b, c, h, w)

        return out + x


class Downsample(nn.Module):
    """Strided convolution for spatial downsampling."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample followed by convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """U-Net noise prediction network for DDPM.

    Args:
        in_channels: Input image channels (3 for RGB).
        base_channels: Base channel count (128 for CIFAR-10).
        channel_mults: Channel multipliers per resolution level.
        num_res_blocks: Number of residual blocks per level.
        attn_resolutions: Resolutions at which to apply self-attention.
        dropout: Dropout rate (0.1 for CIFAR-10).
        image_size: Input image spatial size (for computing resolutions).
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
        image_size=32,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mults)
        time_emb_dim = base_channels * 4

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = [base_channels]
        ch = base_channels
        resolution = image_size

        for level in range(self.num_levels):
            out_ch = base_channels * channel_mults[level]
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResidualBlock(ch, out_ch, time_emb_dim, dropout)])
                if resolution in attn_resolutions:
                    block.append(AttentionBlock(out_ch))
                self.encoder_blocks.append(block)
                ch = out_ch
                channels.append(ch)

            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(ch))
                channels.append(ch)
                resolution //= 2

        # Middle
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, dropout)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            out_ch = base_channels * channel_mults[level]
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = nn.ModuleList(
                    [ResidualBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)]
                )
                if resolution in attn_resolutions:
                    block.append(AttentionBlock(out_ch))
                self.decoder_blocks.append(block)
                ch = out_ch

            if level > 0:
                self.upsamples.append(Upsample(ch))
                resolution *= 2

        # Final output
        self.final_norm = nn.GroupNorm(32, ch)
        self.final_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        time_emb = self.time_mlp(t)

        # Initial conv
        h = self.init_conv(x)
        skips = [h]

        # Encoder
        ds_idx = 0
        block_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                block = self.encoder_blocks[block_idx]
                h = block[0](h, time_emb)
                if len(block) > 1:
                    h = block[1](h)
                skips.append(h)
                block_idx += 1

            if level < self.num_levels - 1:
                h = self.downsamples[ds_idx](h)
                skips.append(h)
                ds_idx += 1

        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder
        us_idx = 0
        block_idx = 0
        for level in reversed(range(self.num_levels)):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                block = self.decoder_blocks[block_idx]
                h = block[0](h, time_emb)
                if len(block) > 1:
                    h = block[1](h)
                block_idx += 1

            if level > 0:
                h = self.upsamples[us_idx](h)
                us_idx += 1

        # Final
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h
