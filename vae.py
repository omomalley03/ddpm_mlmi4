"""
Variational Autoencoder for Latent Diffusion.

Compresses 256×256×3 images to 32×32×4 latents (8× spatial downsampling)
and reconstructs them. Used to enable DDPM in a compact latent space.

Architecture:
    Encoder: 256→128→64→32 with ResBlocks at each resolution
    Decoder: 32→64→128→256 (mirror of encoder)
    Loss: MSE reconstruction + KL divergence (weight 1e-4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple pre-activation residual block (no timestep conditioning)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class Encoder(nn.Module):
    """Encode 256×256×3 images to 32×32×(2*latent_dim) (mean + logvar).

    Three stride-2 downsampling stages: 256→128→64→32.
    """

    def __init__(self, in_channels=3, base_channels=64, latent_dim=4):
        super().__init__()
        ch = base_channels

        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # 256×256 → 128×128
        self.block1a = ResBlock(ch, ch)
        self.block1b = ResBlock(ch, ch)
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

        # 128×128 → 64×64
        self.block2a = ResBlock(ch, ch * 2)
        self.block2b = ResBlock(ch * 2, ch * 2)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)

        # 64×64 → 32×32
        self.block3a = ResBlock(ch * 2, ch * 4)
        self.block3b = ResBlock(ch * 4, ch * 4)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)

        # 32×32 bottleneck
        self.block4a = ResBlock(ch * 4, ch * 4)
        self.block4b = ResBlock(ch * 4, ch * 4)

        # Output: mean and logvar
        self.out_norm = nn.GroupNorm(32, ch * 4)
        self.out_conv = nn.Conv2d(ch * 4, 2 * latent_dim, 3, padding=1)

    def forward(self, x):
        h = self.init_conv(x)

        h = self.block1a(h)
        h = self.block1b(h)
        h = self.down1(h)

        h = self.block2a(h)
        h = self.block2b(h)
        h = self.down2(h)

        h = self.block3a(h)
        h = self.block3b(h)
        h = self.down3(h)

        h = self.block4a(h)
        h = self.block4b(h)

        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        return h


class Decoder(nn.Module):
    """Decode 32×32×latent_dim latents to 256×256×3 images.

    Three upsample stages: 32→64→128→256.
    """

    def __init__(self, out_channels=3, base_channels=64, latent_dim=4):
        super().__init__()
        ch = base_channels

        self.init_conv = nn.Conv2d(latent_dim, ch * 4, 3, padding=1)

        # 32×32 bottleneck
        self.block4a = ResBlock(ch * 4, ch * 4)
        self.block4b = ResBlock(ch * 4, ch * 4)

        # 32×32 → 64×64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1),
        )
        self.block3a = ResBlock(ch * 4, ch * 2)
        self.block3b = ResBlock(ch * 2, ch * 2)

        # 64×64 → 128×128
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
        )
        self.block2a = ResBlock(ch * 2, ch)
        self.block2b = ResBlock(ch, ch)

        # 128×128 → 256×256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.block1a = ResBlock(ch, ch)
        self.block1b = ResBlock(ch, ch)

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.init_conv(z)

        h = self.block4a(h)
        h = self.block4b(h)

        h = self.up3(h)
        h = self.block3a(h)
        h = self.block3b(h)

        h = self.up2(h)
        h = self.block2a(h)
        h = self.block2b(h)

        h = self.up1(h)
        h = self.block1a(h)
        h = self.block1b(h)

        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        return torch.tanh(h)  # output in [-1, 1]


class VAE(nn.Module):
    """Variational Autoencoder.

    Args:
        in_channels: Image channels (3 for RGB).
        base_channels: Base channel count for encoder/decoder.
        latent_dim: Number of latent channels (4).
    """

    def __init__(self, in_channels=3, base_channels=64, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, base_channels, latent_dim)
        self.decoder = Decoder(in_channels, base_channels, latent_dim)

    def encode(self, x):
        """Encode image to latent distribution parameters.

        Returns:
            z: Sampled latent (B, latent_dim, H/8, W/8).
            mu: Mean (B, latent_dim, H/8, W/8).
            logvar: Log variance (B, latent_dim, H/8, W/8).
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z):
        """Decode latent to image."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode → sample → decode.

        Returns:
            x_recon: Reconstructed image (B, C, H, W).
            mu: Latent mean.
            logvar: Latent log variance.
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        """KL divergence from N(mu, sigma) to N(0, I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def recon_loss(x_recon, x):
        """MSE reconstruction loss."""
        return F.mse_loss(x_recon, x)
