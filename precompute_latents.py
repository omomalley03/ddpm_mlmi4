"""
Precompute and cache VAE latents for the entire dataset.

Encodes all images once with the trained VAE encoder, saving the
latent tensors to disk. This avoids running the encoder during
diffusion training.

Output: data/celeba_latents.pt (~480MB for 30K images at 32×32×4 float32)
"""

import os

import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

try:
    import wandb
except ImportError:
    wandb = None

from vae import VAE
from dataset_hires import CelebAHQDataset


SD_VAE_REPO = "stabilityai/sd-vae-ft-ema"


def precompute_latents(
    vae_checkpoint,
    dataset="celeba_hq",
    image_size=256,
    batch_size=32,
    output_path="data/celeba_latents.pt",
    device="cuda",
    data_dir="./data",
    num_workers=4,
    use_stable_diffusion_vae=False,
    use_wandb=False,
    wandb_project="ddpm_mlmi4",
    wandb_run_name=None,
):
    """Encode entire dataset and save latents.

    Args:
        vae_checkpoint: Path to trained VAE checkpoint.
        dataset: Dataset name.
        image_size: Image size.
        batch_size: Encoding batch size.
        output_path: Where to save the latent tensor.
        device: Device string.
        data_dir: Data directory.
        num_workers: DataLoader workers.
        use_stable_diffusion_vae: If True, use Stable Diffusion's VAE instead of the custom one.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (auto-generated if None).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with `pip install wandb` or disable --wandb.")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=dict(
                vae_checkpoint=vae_checkpoint,
                dataset=dataset,
                image_size=image_size,
                batch_size=batch_size,
                output_path=output_path,
                use_stable_diffusion_vae=use_stable_diffusion_vae,
                device=str(device),
            ),
            tags=["precompute"],
        )

    if not use_stable_diffusion_vae:
        # Load VAE
        vae = VAE(in_channels=3, base_channels=64, latent_dim=4).to(device)
        checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["vae"])
        vae.eval()
        print(f"Loaded pre-trained VAE from {vae_checkpoint} (epoch {checkpoint['epoch']})")
    else:
        vae = AutoencoderKL.from_pretrained(SD_VAE_REPO)
        vae = vae.to(device)
        vae.eval()
        print(f"Loaded Stable Diffusion VAE from {SD_VAE_REPO}")

        

    if dataset == "celeba_hq":
        ds = CelebAHQDataset(
            data_dir=os.path.join(data_dir, "celeba_hq"),
            image_size=image_size,
            split="train",
            random_flip=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Encode all images
    n_images = len(ds)
    n_batches = len(dataloader)
    print(f"Encoding {n_images} images...")
    all_latents = []

    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(device)
            if use_stable_diffusion_vae:
                mu = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            else:
                _, mu, _ = vae.encode(x)

            # Use the mean (not sampled z) for more stable training
            all_latents.append(mu.cpu())

            images_done = min((i + 1) * batch_size, n_images)
            if (i + 1) % 50 == 0:
                print(f"  Encoded {images_done}/{n_images}")

            if use_wandb:
                wandb.log({
                    "images_encoded": images_done,
                    "pct_complete": 100.0 * images_done / n_images,
                    "batch": i + 1,
                })

    latents = torch.cat(all_latents, dim=0)
    lat_mean = latents.mean().item()
    lat_std = latents.std().item()
    lat_min = latents.min().item()
    lat_max = latents.max().item()
    print(f"Latent tensor shape: {latents.shape}")
    print(f"Latent stats: mean={lat_mean:.4f}, std={lat_std:.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(latents, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved latents to {output_path} ({size_mb:.1f} MB)")

    if use_wandb:
        wandb.summary.update({
            "latent_shape": str(tuple(latents.shape)),
            "latent_mean": lat_mean,
            "latent_std": lat_std,
            "latent_min": lat_min,
            "latent_max": lat_max,
            "output_size_mb": round(size_mb, 2),
            "n_images": n_images,
        })
        wandb.finish()
