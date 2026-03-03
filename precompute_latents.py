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

from vae import VAE
from dataset_hires import CelebAHQDataset


def precompute_latents(
    vae_checkpoint,
    dataset="celeba_hq",
    image_size=256,
    batch_size=32,
    output_path="data/celeba_latents.pt",
    device="cuda",
    data_dir="./data",
    num_workers=4,
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
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae = VAE(in_channels=3, base_channels=64, latent_dim=4).to(device)
    checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()
    print(f"Loaded VAE from {vae_checkpoint} (epoch {checkpoint['epoch']})")

    # Load dataset (no random flip for deterministic encoding)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset == "celeba_hq":
        from datasets import load_dataset
        hf_dataset = load_dataset(
            "huggan/CelebA-HQ", split="train",
            cache_dir=os.path.join(data_dir, "celeba_hq", "hf_cache"),
        )

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds, tfm):
                self.hf_ds = hf_ds
                self.tfm = tfm
            def __len__(self):
                return len(self.hf_ds)
            def __getitem__(self, idx):
                return self.tfm(self.hf_ds[idx]["image"].convert("RGB"))

        ds = SimpleDataset(hf_dataset, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Encode all images
    print(f"Encoding {len(ds)} images...")
    all_latents = []

    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(device)
            z, mu, _ = vae.encode(x)
            # Use the mean (not sampled z) for more stable training
            all_latents.append(mu.cpu())

            if (i + 1) % 50 == 0:
                print(f"  Encoded {min((i + 1) * batch_size, len(ds))}/{len(ds)}")

    latents = torch.cat(all_latents, dim=0)
    print(f"Latent tensor shape: {latents.shape}")
    print(f"Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(latents, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved latents to {output_path} ({size_mb:.1f} MB)")
