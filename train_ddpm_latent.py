"""
Train a latent DDPM on VAE-encoded OAM images.

Pipeline:
  1. Load trained VAE (128×128, 4-stage: latent 4×8×8).
  2. Encode all OAM images once → TensorDataset of latents.
  3. Train a small UNet DDPM on those latents (in_channels=4, 8×8 spatial).

The latent DDPM UNet uses:
  channel_mults = (1, 2)  → 1 downsampling stage (8→4), avoids spatial collapse
  base_channels = 64
  attn_resolutions = (4,) → attention at the 4×4 bottleneck

Usage:
    python train_ddpm_latent.py \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --mat_path /path/to/data.mat
"""

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE
from dataset_oam import OAMDataset

# VAE config (must match the retrained 128px VAE)
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM = 4
VAE_BASE_CHANNELS = 64
VAE_IMAGE_SIZE = 128
LATENT_SIZE = VAE_IMAGE_SIZE // (2 ** len(VAE_CHANNEL_MULTS))  # 8

# Latent DDPM UNet config
LDM_CHANNEL_MULTS = (1, 2)
LDM_BASE_CHANNELS = 64
LDM_ATN_RES = (LATENT_SIZE // 2,)  # attention at 4×4


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def load_vae(checkpoint_path, device):
    """Load frozen VAE encoder from checkpoint."""
    vae = VAE(
        in_channels=1,
        base_channels=VAE_BASE_CHANNELS,
        channel_mults=VAE_CHANNEL_MULTS,
        latent_dim=VAE_LATENT_DIM,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from epoch {ckpt['epoch']}")
    return vae


@torch.no_grad()
def encode_dataset(vae, mat_path, device, batch_size=64, num_workers=4):
    """Encode all OAM images to latent means (no sampling noise)."""
    dataset = OAMDataset(mat_path, image_size=VAE_IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f"Encoding {len(dataset)} images to latents...")
    all_latents = []
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        _, mu, _ = vae.encode(imgs)   # use mean (deterministic)
        all_latents.append(mu.cpu())
    latents = torch.cat(all_latents, dim=0)
    print(f"Latent tensor: {latents.shape}  "
          f"mean={latents.mean():.3f}  std={latents.std():.3f}")
    return latents


def train(
    vae_checkpoint,
    mat_path,
    batch_size=256,
    lr=2e-4,
    total_steps=200_000,
    save_dir="checkpoints_ldm",
    save_every=50_000,
    log_every=1000,
    resume=None,
    device="cuda",
    num_workers=4,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load VAE and encode dataset
    vae = load_vae(vae_checkpoint, device)
    latents = encode_dataset(vae, mat_path, device,
                             batch_size=batch_size, num_workers=num_workers)
    del vae  # free GPU memory after encoding

    latent_dataset = TensorDataset(latents)
    dataloader = DataLoader(latent_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=True)

    # Latent DDPM UNet: in/out = 4 channels (VAE latent dim)
    model = UNet(
        in_channels=VAE_LATENT_DIM,
        base_channels=LDM_BASE_CHANNELS,
        channel_mults=LDM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=LDM_ATN_RES,
        dropout=0.1,
        image_size=LATENT_SIZE,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Latent DDPM parameters: {param_count:,}")
    print(f"Latent shape: ({VAE_LATENT_DIM}, {LATENT_SIZE}, {LATENT_SIZE})")

    diffusion = GaussianDiffusion(T=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    ema = EMA(model, decay=0.9999)

    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    data_iter = iter(dataloader)
    model.train()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        try:
            (z0,) = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            (z0,) = next(data_iter)

        z0 = z0.to(device)
        t = torch.randint(0, diffusion.T, (z0.shape[0],), device=device)
        loss = diffusion.p_losses(model, z0, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        running_loss += loss.item()

        if (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            print(f"Step {step + 1}/{total_steps} | Loss: {avg_loss:.4f}")
            running_loss = 0.0

        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            ckpt_path = os.path.join(save_dir, f"ldm_ckpt_{step + 1}.pt")
            torch.save({
                "step": step + 1,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vae_checkpoint": vae_checkpoint,
                "latent_size": LATENT_SIZE,
                "latent_dim": VAE_LATENT_DIM,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    print("Latent DDPM training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True,
                        help="Path to trained 128px VAE checkpoint")
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_ldm")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(**vars(args))
