"""
Sample images from a trained DDPM checkpoint.

Loads EMA weights and runs the full reverse process (Algorithm 2).
"""

import os
import torch
from torchvision.utils import save_image

from diffusion import GaussianDiffusion
from model import UNet


def sample(
    checkpoint_path,
    n_samples=64,
    output_dir="samples",
    device="cuda",
    image_size=32,
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
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
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
    shape = (n_samples, 3, image_size, image_size)
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
