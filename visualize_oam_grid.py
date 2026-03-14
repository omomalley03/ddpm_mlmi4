"""
Visualise OAM dataset: one example per (mode × turbulence strength) combination.

Usage:
    python visualize_oam_grid.py --mat_path /path/to/data.mat
    python visualize_oam_grid.py --mat_path /path/to/data.mat --out grid.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_oam import OAMDataset, MODE_DISPLAY

ALL_MODES = ["gauss", "p1", "p2", "p3", "p4", "n1", "n2", "n3"]


def make_grid(mat_path, out_path="oam_grid.png"):
    dataset = OAMDataset(mat_path, modes=ALL_MODES, normalize=True)

    modes = dataset.modes
    turb_cats = dataset.turb_categories  # e.g. [0, 1, 2, 3, ...]
    n_modes = len(modes)
    n_turb = len(turb_cats)

    # Build lookup: (mode_idx, turb_label) -> first matching dataset index
    lookup = {}
    for i in range(len(dataset)):
        key = (dataset.mode_labels[i], dataset.turb_labels[i])
        if key not in lookup:
            lookup[key] = i

    fig, axes = plt.subplots(
        n_modes, n_turb,
        figsize=(n_turb * 1.4, n_modes * 1.6),
        squeeze=False,
    )

    for row, mode_idx in enumerate(range(n_modes)):
        for col, turb in enumerate(turb_cats):
            ax = axes[row, col]
            key = (mode_idx, turb)
            if key in lookup:
                img, _, _ = dataset[lookup[key]]
                # [-1, 1] → [0, 1] for display
                img_np = ((img.squeeze().numpy() + 1) / 2).clip(0, 1)
                ax.imshow(img_np, cmap="hot", vmin=0, vmax=1)
            else:
                ax.set_facecolor("black")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"T={turb}", fontsize=7, pad=2)

        axes[row, 0].set_ylabel(
            MODE_DISPLAY.get(modes[mode_idx], modes[mode_idx]),
            fontsize=8, rotation=0, labelpad=40, va="center",
        )

    plt.suptitle("OAM modes × turbulence strength", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", required=True, help="Path to the .mat data file")
    parser.add_argument("--out", default="oam_grid.png", help="Output image path")
    args = parser.parse_args()
    make_grid(args.mat_path, args.out)
