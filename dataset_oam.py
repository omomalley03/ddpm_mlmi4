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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io
import h5py


# Modes to use (subset of: gauss, p1, p2, p3, p4, n1, n2, n3)
MODES = ["gauss", "p4"]

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


def _is_hdf5(mat_path):
    try:
        scipy.io.loadmat(mat_path, variable_names=[])
        return False
    except NotImplementedError:
        return True


def _read_labels(mat_path, mode, is_hdf5):
    """Read only the labels array for one mode (small — always fits in RAM)."""
    key = f"{mode}_labels"
    if is_hdf5:
        with h5py.File(mat_path, "r") as f:
            return f[key][()].flatten().astype(np.int64)
    else:
        mat = scipy.io.loadmat(mat_path, variable_names=[key])
        return mat[key].flatten().astype(np.int64)


class OAMDataset(Dataset):
    """Lazy OAM dataset — reads one image at a time from disk, never loads all at once.

    Args:
        mat_path: Path to the .mat file.
        modes: List of modes to include (default: MODES).
        image_size: Resize spatial dims to this size. None = keep original (320).
        normalize: If True, normalize each image to [-1, 1] via per-image max.
    """

    def __init__(self, mat_path, modes=None, image_size=None, normalize=True):
        self.mat_path   = mat_path
        self.modes      = modes or MODES
        self.image_size = image_size
        self.normalize  = normalize
        self._hdf5      = _is_hdf5(mat_path)
        self._file      = None  # opened lazily per worker in __getitem__

        # Load only labels (integers, negligible RAM) to build the sample index
        mode_labels_list = []
        turb_labels_list = []
        self._cumsize = [0]

        for mode_idx, mode in enumerate(self.modes):
            labels = _read_labels(mat_path, mode, self._hdf5)
            n = len(labels)
            mode_labels_list.append(np.full(n, mode_idx, dtype=np.int64))
            turb_labels_list.append(labels)
            self._cumsize.append(self._cumsize[-1] + n)

        self.mode_labels     = np.concatenate(mode_labels_list)
        self.turb_labels     = np.concatenate(turb_labels_list)
        self.turb_categories = sorted(np.unique(self.turb_labels).tolist())

        print(f"OAMDataset: {len(self)} samples | modes={self.modes} | hdf5={self._hdf5}")
        print(f"  Turbulence categories: {self.turb_categories}")

    # Keep file handle out of pickle (DataLoader worker forking)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __len__(self):
        return self._cumsize[-1]

    def _global_to_local(self, idx):
        mode_idx  = int(np.searchsorted(self._cumsize[1:], idx, side="right"))
        local_idx = idx - self._cumsize[mode_idx]
        return mode_idx, local_idx

    def _read_image(self, mode, local_idx):
        """Read a single (H, W) float32 image without loading the full array."""
        key = f"{mode}_X"
        if self._hdf5:
            # h5py stores MATLAB (H,W,1,N) in reversed C order → raw shape (N,1,W,H)
            # Slice [local_idx, 0] → (W, H), then .T → (H, W)
            if self._file is None:
                self._file = h5py.File(self.mat_path, "r")
            return self._file[key][local_idx, 0].astype(np.float32).T
        else:
            # Legacy .mat: scipy loads the full array (only viable for small files)
            mat = scipy.io.loadmat(self.mat_path, variable_names=[key])
            return mat[key][:, :, 0, local_idx].astype(np.float32)  # (H, W)

    def __getitem__(self, idx):
        mode_idx, local_idx = self._global_to_local(idx)
        img_np = self._read_image(self.modes[mode_idx], local_idx)   # (H, W)
        img = torch.from_numpy(img_np).unsqueeze(0)                  # (1, H, W)

        if self.image_size is not None:
            img = F.interpolate(img.unsqueeze(0), size=self.image_size,
                                mode="bilinear", align_corners=False).squeeze(0)

        if self.normalize:
            vmax = img.max()
            if vmax > 0:
                img = img / vmax
            img = img * 2.0 - 1.0  # [0,1] → [-1,1]

        return (
            img,
            torch.tensor(self.mode_labels[idx], dtype=torch.long),
            torch.tensor(self.turb_labels[idx], dtype=torch.long),
        )

    def mode_name(self, mode_idx):
        return self.modes[mode_idx]

    def mode_display_name(self, mode_idx):
        return MODE_DISPLAY.get(self.modes[mode_idx], self.modes[mode_idx])


def get_oam_dataloader(mat_path, batch_size=32, modes=None, image_size=None,
                       num_workers=0, shuffle=True):
    """Create a DataLoader for OAM beam images.

    num_workers=0 by default: h5py file handles don't fork safely across workers.
    """
    dataset = OAMDataset(mat_path, modes=modes, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    ), dataset
