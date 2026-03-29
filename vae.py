"""
Variational Autoencoder for Latent Diffusion / Latent Space Visualization.

Supports arbitrary input sizes and depths via channel_mults.

Examples:
    OAM 320x320 grayscale   -> VAE(in_channels=1, channel_mults=(1,2,4,4)) -> 20x20x8 latent

Loss: MSE reconstruction + KL divergence (kl_weight=1e-4 default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-activation residual block (no timestep conditioning)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # GroupNorm requires num_groups to divide num_channels
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class Encoder(nn.Module):
    """Encode images to latent distribution parameters (mean + logvar).

    Args:
        in_channels: Input image channels.
        base_channels: Base channel count.
        channel_mults: Multipliers per downsampling stage.
        latent_dim: Number of latent channels.
    """

    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4), latent_dim=4):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels

        for mult in channel_mults:
            out_ch = base_channels * mult
            self.stages.append(nn.Sequential(ResBlock(ch, out_ch), ResBlock(out_ch, out_ch)))
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        self.bottleneck = nn.Sequential(ResBlock(ch, ch), ResBlock(ch, ch))
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, 2 * latent_dim, 3, padding=1)

    def forward(self, x):
        h = self.init_conv(x)
        for stage, down in zip(self.stages, self.downsamples):
            h = stage(h)
            h = down(h)
        h = self.bottleneck(h)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class Decoder(nn.Module):
    """Decode latents back to images.

    Args:
        out_channels: Output image channels.
        base_channels: Base channel count (must match Encoder).
        channel_mults: Same tuple as Encoder (applied in reverse).
        latent_dim: Number of latent channels.
    """

    def __init__(self, out_channels=3, base_channels=64, channel_mults=(1, 2, 4), latent_dim=4):
        super().__init__()
        ch = base_channels * channel_mults[-1]
        self.init_conv = nn.Conv2d(latent_dim, ch, 3, padding=1)
        self.bottleneck = nn.Sequential(ResBlock(ch, ch), ResBlock(ch, ch))

        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            ))
            self.stages.append(nn.Sequential(ResBlock(ch, out_ch), ResBlock(out_ch, out_ch)))
            ch = out_ch

        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.init_conv(z)
        h = self.bottleneck(h)
        for up, stage in zip(self.upsamples, self.stages):
            h = up(h)
            h = stage(h)
        h = F.silu(self.out_norm(h))
        return torch.tanh(self.out_conv(h))  # output in [-1, 1]


class VAE(nn.Module):
    """Variational Autoencoder.

    Args:
        in_channels: Image channels (3 for RGB, 1 for grayscale).
        base_channels: Base channel count.
        channel_mults: Multipliers per downsampling stage. Length = number of stages.
            (1,2,4)   -> 3 stages, 8x  spatial compression  (256->32)
            (1,2,4,4) -> 4 stages, 16x spatial compression  (320->20)
        latent_dim: Number of latent channels.
    """

    def __init__(self, in_channels=3, base_channels=64, channel_mults=(1, 2, 4), latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_mults = channel_mults
        self.encoder = Encoder(in_channels, base_channels, channel_mults, latent_dim)
        self.decoder = Decoder(in_channels, base_channels, channel_mults, latent_dim)

    def encode(self, x):
        """Returns (z_sampled, mu, logvar)."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Returns (x_recon, mu, logvar)."""
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def recon_loss(x_recon, x):
        return F.mse_loss(x_recon, x)
