"""
Generate 256×256 images via Latent Diffusion.

1. Run DDPM reverse process to generate 32×32×4 latents
2. Decode latents to 256×256×3 images using the trained VAE decoder
"""

import os
import torch
from torchvision.utils import save_image, make_grid

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE


def sample_latent(
    diffusion_checkpoint,
    vae_checkpoint,
    n_samples=16,
    output_dir="samples_latent",
    device="cuda",
):
    """Generate high-resolution images via latent diffusion.

    Args:
        diffusion_checkpoint: Path to trained latent diffusion checkpoint.
        vae_checkpoint: Path to trained VAE checkpoint.
        n_samples: Number of images to generate.
        output_dir: Directory to save generated images.
        device: Device string.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load diffusion model (UNet with 4 latent channels)
    model = UNet(
        in_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=32,
    ).to(device)

    diff_ckpt = torch.load(diffusion_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(diff_ckpt["ema"])
    model.eval()
    print(f"Loaded diffusion model from step {diff_ckpt['step']}")

    # Load VAE decoder
    vae = VAE(in_channels=3, base_channels=64, latent_dim=4).to(device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(vae_ckpt["vae"])
    vae.eval()
    print(f"Loaded VAE from epoch {vae_ckpt['epoch']}")

    # Generate latents
    diffusion = GaussianDiffusion(T=1000, device=device)
    print(f"Generating {n_samples} latents (32×32×4)...")
    shape = (n_samples, 4, 32, 32)
    latents = diffusion.p_sample_loop(model, shape)

    # Decode to images
    print("Decoding latents to 256×256 images...")
    with torch.no_grad():
        images = vae.decode(latents)

    # Denormalize [-1,1] → [0,1]
    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    # Save grid
    step = diff_ckpt["step"]
    grid_path = os.path.join(output_dir, f"latent_grid_step{step}.png")
    save_image(images, grid_path, nrow=int(n_samples**0.5), padding=2)
    print(f"Saved image grid: {grid_path}")

    # Save individual images
    for i in range(n_samples):
        img_path = os.path.join(output_dir, f"latent_sample_{i:04d}.png")
        save_image(images[i], img_path)

    print("Latent diffusion sampling complete.")
