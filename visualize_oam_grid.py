"""
Visualise OAM dataset: first and last image for each mode.

Usage:
    python visualize_oam_grid.py --mat_path /path/to/data.mat
    python visualize_oam_grid.py --mat_path /path/to/data.mat --out grid.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import scipy.io

from dataset_oam import MODE_DISPLAY

ALL_MODES = ["gauss", "p1", "p2", "p3", "p4", "n1", "n2", "n3"]


def load_two_images(mat_path, mode):
    """Return (first_img, last_img) as (H, W) float32 arrays, normalized to [0,1]."""
    key = f"{mode}_X"
    try:
        mat = scipy.io.loadmat(mat_path, variable_names=[key])
        arr = mat[key]  # (H, W, 1, N)
        first = arr[:, :, 0, 0].astype(np.float32)
        last  = arr[:, :, 0, -1].astype(np.float32)
    except NotImplementedError:
        # HDF5 / v7.3 — h5py stores dims reversed; shape is (N, 1, W, H)
        with h5py.File(mat_path, "r") as f:
            ds = f[key]
            first = ds[0,  0].astype(np.float32).T   # (H, W)
            last  = ds[-1, 0].astype(np.float32).T

    def norm(img):
        vmax = img.max()
        return img / vmax if vmax > 0 else img

    return norm(first), norm(last)


def make_grid(mat_path, out_path="oam_grid.png"):
    _, axes = plt.subplots(len(ALL_MODES), 2, figsize=(3, len(ALL_MODES) * 1.6), squeeze=False)

    for row, mode in enumerate(ALL_MODES):
        first, last = load_two_images(mat_path, mode)
        for col, img in enumerate([first, last]):
            axes[row, col].imshow(img, cmap="hot", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            MODE_DISPLAY.get(mode, mode),
            fontsize=8, rotation=0, labelpad=40, va="center",
        )

    axes[0, 0].set_title("First", fontsize=8)
    axes[0, 1].set_title("Last", fontsize=8)

    plt.suptitle("OAM modes — first & last image", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--out", default="oam_grid.png")
    args = parser.parse_args()
    make_grid(args.mat_path, args.out)
