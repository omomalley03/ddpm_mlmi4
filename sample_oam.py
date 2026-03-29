"""
Sample images from a trained DDPM checkpoint.

Loads EMA weights and runs the full reverse process (Algorithm 2).
"""

import os
import torch
from torchvision.utils import save_image, make_grid

from diffusion import GaussianDiffusion
from model import UNet
OAM_CHANNEL_MULTS = (1, 2, 4, 4, 4)
OAM_LATENT_DIM = 4
OAM_BASE_CHANNELS = 64

def sample(
    checkpoint_path,
    n_samples=64,
    output_dir="samples",
    device="cuda",
    image_size=128,
):
    """Generate images from a trained DDPM model.

    Args:
        checkpoint_path: Path to saved checkpoint.
        n_samples: Number of images to generate.
        output_dir: Directory to save generated images.
        device: Device string.
        image_size: Image spatial size.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,  # No dropout at inference
        image_size=image_size,
    ).to(device)

    # Load EMA weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Diffusion
    diffusion = GaussianDiffusion(T=1000, device=device)

    # Generate
    print(f"Generating {n_samples} samples...")
    shape = (n_samples, 1, image_size, image_size)
    images = diffusion.p_sample_loop(model, shape)

    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    # Save grid
    grid_path = os.path.join(output_dir, f"grid_step{checkpoint['step']}.png")
    save_image(images, grid_path, nrow=int(n_samples**0.5), padding=2)
    print(f"Saved image grid: {grid_path}")

    # Save individual images
    for i in range(n_samples):
        img_path = os.path.join(output_dir, f"sample_{i:04d}.png")
        save_image(images[i], img_path)

    print("Sampling complete.")


def sample_progression(
    checkpoint_path,
    n_samples=4,
    n_frames=10,
    output_dir="samples",
    device="cuda",
    image_size=128,
):
    """Visualise the denoising process as a grid.

    Output grid: each row is one sample, each column is a timestep.
    Left = pure noise (t=T), right = final image (t=0).

    Args:
        checkpoint_path: Path to saved checkpoint.
        n_samples: Number of independent denoising trajectories to show (rows).
        n_frames: Number of timesteps to capture across the trajectory (columns).
        output_dir: Directory to save the grid image.
        device: Device string.
        image_size: Image spatial size.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=image_size,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    print(f"Loaded checkpoint from step {checkpoint['step']}")

    diffusion = GaussianDiffusion(T=1000, device=device)

    print(f"Generating {n_samples} denoising trajectories ({n_frames} frames each)...")
    shape = (n_samples, 1, image_size, image_size)
    frames = diffusion.p_sample_loop_progressive(model, shape, n_frames=n_frames)

    # frames[i]: tensor (n_samples, 1, H, W) — one per saved timestep
    # Stack to (n_samples, n_frames, 1, H, W), then flatten to (n_samples*n_frames, 1, H, W)
    # make_grid with nrow=n_frames gives one row per sample
    stacked = torch.stack(frames, dim=1)                          # (n_samples, n_frames, 1, H, W)
    stacked = stacked.view(-1, 1, image_size, image_size)         # (n_samples * n_frames, 1, H, W)

    # Denormalize from [-1, 1] to [0, 1]
    stacked = (stacked + 1.0) / 2.0
    stacked = stacked.clamp(0.0, 1.0)

    grid = make_grid(stacked, nrow=n_frames, padding=2)
    out_path = os.path.join(output_dir, f"progression_step{checkpoint['step']}.png")
    save_image(grid, out_path)
    print(f"Saved denoising progression grid: {out_path}")
    print(f"  Grid layout: {n_samples} rows (samples) x {n_frames} columns (timesteps)")
