"""
Sample from the Latent DDPM and decode with the frozen VAE.

Pipeline:
  1. Sample latent tensors (4x8x8) using the trained latent DDPM.
  2. Decode each latent with the frozen 128px VAE decoder -> 128x128 images.
  3. Save a sample grid and compare side-by-side with pixel-DDPM samples
     (if --pixel_checkpoint is provided).

Usage:
    python sample_ldm.py \\
        --ldm_checkpoint checkpoints_ldm/ldm_ckpt_200000.pt \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --output_dir samples_ldm

    # Side-by-side comparison:
    python sample_ldm.py \\
        --ldm_checkpoint checkpoints_ldm/ldm_ckpt_200000.pt \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --pixel_checkpoint checkpoints_ddpm_oam/ckpt_300000.pt \\
        --output_dir samples_ldm
"""

import argparse
import os

import torch
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE

# VAE config — must match retrained 128px VAE
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM = 4
VAE_BASE_CHANNELS = 64
VAE_IMAGE_SIZE = 128
LATENT_SIZE = VAE_IMAGE_SIZE // (2 ** len(VAE_CHANNEL_MULTS))  # 8

# Latent DDPM UNet config
LDM_CHANNEL_MULTS = (1, 2)
LDM_BASE_CHANNELS = 64
LDM_ATN_RES = (LATENT_SIZE // 2,)  # 4

# Pixel DDPM config (for comparison)
PIXEL_CHANNEL_MULTS = (1, 2, 4, 4, 4)
PIXEL_BASE_CHANNELS = 64


def load_vae(checkpoint_path, device):
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


def load_ldm(checkpoint_path, device):
    model = UNet(
        in_channels=VAE_LATENT_DIM,
        base_channels=LDM_BASE_CHANNELS,
        channel_mults=LDM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=LDM_ATN_RES,
        dropout=0.0,
        image_size=LATENT_SIZE,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded latent DDPM from step {ckpt['step']}")
    return model, ckpt["step"]


def load_pixel_ddpm(checkpoint_path, device, image_size=128):
    model = UNet(
        in_channels=1,
        base_channels=PIXEL_BASE_CHANNELS,
        channel_mults=PIXEL_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=image_size,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded pixel DDPM from step {ckpt['step']}")
    return model


def main(
    ldm_checkpoint,
    vae_checkpoint,
    output_dir="samples_ldm",
    n_samples=64,
    pixel_checkpoint=None,
    image_size=128,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    vae = load_vae(vae_checkpoint, device)
    ldm_model, ldm_step = load_ldm(ldm_checkpoint, device)
    diffusion = GaussianDiffusion(T=1000, device=device)

    # --- LDM samples ---
    print(f"Sampling {n_samples} latents from latent DDPM...")
    latent_shape = (n_samples, VAE_LATENT_DIM, LATENT_SIZE, LATENT_SIZE)
    with torch.no_grad():
        latents = diffusion.p_sample_loop(ldm_model, latent_shape)
        ldm_images = vae.decode(latents)        # (N, 1, 128, 128) in [-1, 1]

    ldm_vis = ((ldm_images + 1.0) / 2.0).clamp(0, 1)

    grid_path = os.path.join(output_dir, f"ldm_samples_step{ldm_step}.png")
    save_image(ldm_vis, grid_path, nrow=int(n_samples ** 0.5), padding=2)
    print(f"Saved LDM sample grid: {grid_path}")

    # Save individual images
    for i in range(n_samples):
        save_image(ldm_vis[i], os.path.join(output_dir, f"ldm_{i:04d}.png"))

    # --- Side-by-side comparison with pixel DDPM ---
    if pixel_checkpoint is not None:
        pixel_model = load_pixel_ddpm(pixel_checkpoint, device, image_size)
        pixel_shape = (n_samples, 1, image_size, image_size)
        print(f"Sampling {n_samples} images from pixel DDPM...")
        with torch.no_grad():
            pixel_images = diffusion.p_sample_loop(pixel_model, pixel_shape)
        pixel_vis = ((pixel_images + 1.0) / 2.0).clamp(0, 1)

        # Build comparison figure: top half = pixel DDPM, bottom half = LDM
        nrow = int(n_samples ** 0.5)
        pixel_grid = make_grid(pixel_vis, nrow=nrow, padding=2)
        ldm_grid = make_grid(ldm_vis, nrow=nrow, padding=2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(nrow * 2, nrow * 4))
        ax1.imshow(pixel_grid.permute(1, 2, 0).cpu().numpy(), cmap="hot")
        ax1.set_title("Pixel-space DDPM samples", fontsize=12)
        ax1.axis("off")
        ax2.imshow(ldm_grid.permute(1, 2, 0).cpu().numpy(), cmap="hot")
        ax2.set_title("LDM samples (latent DDPM + VAE decode)", fontsize=12)
        ax2.axis("off")
        plt.tight_layout()
        compare_path = os.path.join(output_dir, "comparison_pixel_vs_ldm.png")
        plt.savefig(compare_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison grid: {compare_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_checkpoint", required=True,
                        help="Path to trained latent DDPM checkpoint")
    parser.add_argument("--vae_checkpoint", required=True,
                        help="Path to trained 128px VAE checkpoint")
    parser.add_argument("--output_dir", default="samples_ldm")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--pixel_checkpoint", default=None,
                        help="Optional pixel-DDPM checkpoint for side-by-side comparison")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(**vars(args))
