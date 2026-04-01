"""
High-resolution dataset loading for Latent Diffusion.

Supports local CelebA-HQ image folders first, with Hugging Face as a fallback.
Images are normalized to [-1, 1].
"""

import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _default_celeba_hq_candidates(data_dir):
    """Return candidate local directories for CelebA-HQ images."""
    candidates = [
        os.environ.get("CELEBA_HQ_DIR"),
        os.path.join(data_dir, "celeba_hq_256"),
        data_dir,
    ]

    kagglehub_root = os.path.expanduser(
        "~/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions"
    )
    if os.path.isdir(kagglehub_root):
        version_dirs = sorted(glob.glob(os.path.join(kagglehub_root, "*", "celeba_hq_256")))
        candidates.extend(version_dirs)

    return [path for path in candidates if path]


def _find_local_celeba_hq_dir(data_dir):
    """Find a local image directory containing CelebA-HQ files."""
    for candidate in _default_celeba_hq_candidates(data_dir):
        if not os.path.isdir(candidate):
            continue

        has_images = any(
            glob.glob(os.path.join(candidate, pattern))
            for pattern in ("*.jpg", "*.jpeg", "*.png")
        )
        if has_images:
            return candidate

    return None


class CelebAHQDataset(Dataset):
    """CelebA-HQ dataset loaded from local files or Hugging Face.

    Prefers local image folders when available.
    """

    def __init__(self, data_dir="./data/celeba_hq", image_size=256, split="train", random_flip=True):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] → [-1,1]
        ])

        self.local_dir = _find_local_celeba_hq_dir(data_dir)
        self.image_paths = None
        self.dataset = None

        if self.local_dir is not None:
            self.image_paths = []
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                self.image_paths.extend(glob.glob(os.path.join(self.local_dir, pattern)))
            self.image_paths = sorted(self.image_paths)
            if not self.image_paths:
                raise RuntimeError(f"No images found in local CelebA-HQ directory: {self.local_dir}")
            print(f"Loaded CelebA-HQ from local directory: {self.local_dir} ({len(self.image_paths)} images)")
        else:
            raise RuntimeError(
                "No local CelebA-HQ images found. "
                "Set the CELEBA_HQ_DIR environment variable to the directory containing your .jpg/.png images, "
                "or place them under data/celeba_hq_256/. "
                f"Searched: {_default_celeba_hq_candidates(data_dir)}"
            )

    def __len__(self):
        if self.image_paths is not None:
            return len(self.image_paths)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.image_paths is not None:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        else:
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
                             image_size=image_size,
                             random_flip=True)
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
