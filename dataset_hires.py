"""
High-resolution dataset loading for Latent Diffusion.

CelebA-HQ 256×256: downloaded via HuggingFace datasets, normalized to [-1, 1].
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebAHQDataset(Dataset):
    """CelebA-HQ dataset loaded from HuggingFace.

    Downloads ~1GB on first use, then caches locally.
    """

    def __init__(self, data_dir="./data/celeba_hq", image_size=256, split="train"):
        from datasets import load_dataset

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
        ])

        cache_dir = os.path.join(data_dir, "hf_cache")
        self.dataset = load_dataset(
            "huggan/CelebA-HQ",
            split=split,
            cache_dir=cache_dir,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        return self.transform(image)


class LatentDataset(Dataset):
    """Dataset of precomputed latent tensors."""

    def __init__(self, path="data/celeba_latents.pt"):
        self.latents = torch.load(path, weights_only=True)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]


def get_hires_dataloader(dataset="celeba_hq", image_size=256, batch_size=16,
                         data_dir="./data", num_workers=4):
    """Create a DataLoader for high-resolution images."""
    if dataset == "celeba_hq":
        ds = CelebAHQDataset(data_dir=os.path.join(data_dir, "celeba_hq"),
                             image_size=image_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


def get_latent_dataloader(latent_path, batch_size=128, num_workers=4):
    """Create a DataLoader for precomputed latents."""
    ds = LatentDataset(latent_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
