"""
Dataset loading for DDPM training.

CIFAR-10: auto-downloaded via torchvision, normalized to [-1, 1],
with RandomHorizontalFlip augmentation (per Ho et al. 2020).
"""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloader(dataset="cifar10", batch_size=128, data_dir="./data", num_workers=4, subset_size=None):
    """Create a DataLoader for training.

    Args:
        dataset: Dataset name (currently only 'cifar10').
        batch_size: Training batch size.
        data_dir: Directory to store/load data.
        num_workers: Number of data loading workers.

    Returns:
        DataLoader instance.
    """
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] -> [-1,1]
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if subset_size is not None:
        train_dataset = Subset(train_dataset, range(subset_size))

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataloader
