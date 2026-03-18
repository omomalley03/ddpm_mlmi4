"""
Dataset loader for OAM laser beam images stored in a .mat file.

File structure:
    Each mode has two variables:
        <mode>_X      : float array, shape (320, 320, 1, N) — intensity images
        <mode>_labels : int array,   shape (N, 1) or (1, N) — turbulence strength categories

    Modes: gauss, p1, p2, p3, p4, n1, n2, n3

Usage:
    dataset = OAMDataset("/path/to/data.mat")
    loader  = get_oam_dataloader("/path/to/data.mat", batch_size=32)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import h5py


def _load_mat(mat_path, keys=None):
    """Load a .mat file, supporting both legacy (<v7.3) and HDF5 (v7.3) formats.

    Args:
        keys: Optional list of variable names to load. If None, loads everything.
              Use this to avoid reading the full file when only a subset is needed.
    """
    try:
        mat = scipy.io.loadmat(mat_path)
        if keys is not None:
            return {k: mat[k] for k in keys if k in mat}
        return mat
    except NotImplementedError:
        # MATLAB v7.3 (HDF5) — h5py returns arrays in C order (dims reversed vs MATLAB).
        # Transpose each array so shapes match what scipy.io.loadmat would return.
        data = {}
        with h5py.File(mat_path, "r") as f:
            target_keys = keys if keys is not None else list(f.keys())
            for key in target_keys:
                if key in f and isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][()].T
        return data


# Modes to use (subset of: gauss, p1, p2, p3, p4, n1, n2, n3)
MODES = ["gauss", "p1", "p2", "p3", "p4"]

# Human-readable labels for plotting
MODE_DISPLAY = {
    "gauss": "Gaussian",
    "p1": "OAM ell=+1",
    "p2": "OAM ell=+2",
    "p3": "OAM ell=+3",
    "p4": "OAM ell=+4",
    "n1": "OAM ell=-1",
    "n2": "OAM ell=-2",
    "n3": "OAM ell=-3",
}


class OAMDataset(Dataset):
    """OAM beam intensity images with mode and turbulence labels.

    Args:
        mat_path: Path to the .mat file.
        modes: List of modes to include (default: all 8).
        image_size: Resize images to this size. None = keep original (320).
        normalize: If True, normalize each image to [-1, 1] via per-image max.
    """

    def __init__(self, mat_path, modes=None, image_size=None, normalize=True,
                 turb_levels=None):
        """
        Args:
            turb_levels: List of turbulence label values to include (e.g. [1, 2, 3]).
                         None = include all levels.
        """
        if modes is None:
            modes = MODES

        needed_keys = [f"{m}_X" for m in modes] + [f"{m}_labels" for m in modes]
        data = _load_mat(mat_path, keys=needed_keys)
        self._check_keys(data, modes)

        images_list = []
        mode_labels_list = []
        turb_labels_list = []

        for mode_idx, mode in enumerate(modes):
            x_key = f"{mode}_X"
            l_key = f"{mode}_labels"

            # Shape: (H, W, 1, N) → (N, 1, H, W)
            imgs = data[x_key]  # (320, 320, 1, N)
            imgs = imgs.transpose(3, 2, 0, 1).astype(np.float32)  # (N, 1, 320, 320)

            # Turbulence labels: flatten to (N,)
            labels = data[l_key].flatten().astype(np.int64)

            # Filter by turbulence level if requested
            if turb_levels is not None:
                mask = np.isin(labels, turb_levels)
                imgs = imgs[mask]
                labels = labels[mask]

            images_list.append(imgs)
            mode_labels_list.append(np.full(len(imgs), mode_idx, dtype=np.int64))
            turb_labels_list.append(labels)

        self.images = np.concatenate(images_list, axis=0)            # (total_N, 1, H, W)
        self.mode_labels = np.concatenate(mode_labels_list, axis=0)  # (total_N,)
        self.turb_labels = np.concatenate(turb_labels_list, axis=0)  # (total_N,)
        self.modes = modes
        self.image_size = image_size
        self.normalize = normalize

        # Unique turbulence categories (sorted)
        self.turb_categories = sorted(np.unique(self.turb_labels).tolist())

        print(f"Loaded {len(self.images)} images from {len(modes)} modes")
        print(f"  Modes: {modes}")
        print(f"  Image shape: {self.images.shape[1:]}")
        print(f"  Turbulence categories: {self.turb_categories}")

    @staticmethod
    def _check_keys(data, modes):
        missing = [f"{m}_X" for m in modes if f"{m}_X" not in data]
        if missing:
            available = [k for k in data.keys() if not k.startswith("_")]
            raise KeyError(f"Keys not found in .mat file: {missing}. Available: {available}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])  # (1, H, W)

        if self.image_size is not None:
            import torch.nn.functional as F
            img = F.interpolate(img.unsqueeze(0), size=self.image_size, mode="bilinear",
                                align_corners=False).squeeze(0)

        if self.normalize:
            # Per-image normalization: [0, max] → [-1, 1]
            vmax = img.max()
            if vmax > 0:
                img = img / vmax           # [0, 1]
            img = img * 2.0 - 1.0         # [-1, 1]

        mode_label = torch.tensor(self.mode_labels[idx], dtype=torch.long)
        turb_label = torch.tensor(self.turb_labels[idx], dtype=torch.long)

        return img, mode_label, turb_label

    def mode_name(self, mode_idx):
        return self.modes[mode_idx]

    def mode_display_name(self, mode_idx):
        return MODE_DISPLAY.get(self.modes[mode_idx], self.modes[mode_idx])


def get_oam_dataloader(mat_path, batch_size=32, modes=None, image_size=None,
                       num_workers=0, shuffle=True, turb_levels=None):
    """Create a DataLoader for OAM beam images."""
    dataset = OAMDataset(mat_path, modes=modes, image_size=image_size,
                         turb_levels=turb_levels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    ), dataset
