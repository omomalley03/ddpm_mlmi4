"""
Evaluate reconstruction quality of the retrained 128px VAE.

For each (mode, turbulence_level) cell:
  - Encode real images → decode → measure MSE and SSIM
  - Save reconstruction grid (real vs recon side-by-side)
  - Save quality_table.csv

Usage:
    python analyse_vae_quality.py \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --mat_path /path/to/data.mat \\
        --output_dir analysis_vae_quality
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae import VAE
from dataset_oam import OAMDataset

# Must match the retrained 128px VAE
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM = 4
VAE_BASE_CHANNELS = 64
VAE_IMAGE_SIZE = 128


def ssim_simple(a, b):
    """Simple SSIM approximation (no windowing) for single-channel images in [-1,1]."""
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = ((a - mu_a) ** 2).mean()
    sig_b = ((b - mu_b) ** 2).mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
    return float(num / den)


def main(vae_checkpoint, mat_path, output_dir, n_per_cell=8, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load VAE
    vae = VAE(
        in_channels=1,
        base_channels=VAE_BASE_CHANNELS,
        channel_mults=VAE_CHANNEL_MULTS,
        latent_dim=VAE_LATENT_DIM,
    ).to(device)
    ckpt = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    print(f"Loaded VAE from epoch {ckpt['epoch']}")

    dataset = OAMDataset(mat_path, image_size=VAE_IMAGE_SIZE)
    modes = dataset.modes
    # Collect unique turbulence labels
    turb_labels_all = dataset.turb_labels
    unique_turbs = sorted(set(int(t) for t in turb_labels_all))

    print(f"Modes: {modes}  |  Turb levels: {unique_turbs}")

    rows = []  # for CSV

    # --- Reconstruction grid ---
    # Layout: rows = modes, columns = turb levels, each cell = real|recon pair
    n_modes = len(modes)
    n_turbs = len(unique_turbs)
    fig_cols = n_per_cell * 2  # real + recon per turb level per column group
    fig, axes = plt.subplots(
        n_modes * 2, n_turbs * n_per_cell,
        figsize=(n_turbs * n_per_cell * 1.5, n_modes * 4),
    )
    # axes shape: (n_modes*2, n_turbs * n_per_cell)
    if axes.ndim == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for mi, mode in enumerate(modes):
            for ti, turb in enumerate(unique_turbs):
                # Find samples matching this (mode, turb)
                mode_idx = mi
                mask = (
                    (np.array(dataset.mode_labels) == mode_idx) &
                    (np.array(dataset.turb_labels) == turb)
                )
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    rows.append({"mode": mode, "turb": turb, "mse": float("nan"), "ssim": float("nan")})
                    continue

                rng = np.random.default_rng(42)
                chosen = rng.choice(indices, size=min(n_per_cell, len(indices)), replace=False)

                mses, ssims = [], []
                for j, idx in enumerate(chosen):
                    img, _, _ = dataset[idx]
                    img_t = img.unsqueeze(0).to(device)
                    recon_t, _, _ = vae(img_t)

                    img_np = img.squeeze().cpu().numpy()
                    recon_np = recon_t.squeeze().cpu().numpy()

                    mse = float(((img_np - recon_np) ** 2).mean())
                    s = ssim_simple(img_np, recon_np)
                    mses.append(mse)
                    ssims.append(s)

                    # Fill grid
                    col = ti * n_per_cell + j
                    vis_orig = ((img_np + 1) / 2).clip(0, 1)
                    vis_recon = ((recon_np + 1) / 2).clip(0, 1)
                    axes[mi * 2, col].imshow(vis_orig, cmap="hot", vmin=0, vmax=1)
                    axes[mi * 2, col].axis("off")
                    axes[mi * 2 + 1, col].imshow(vis_recon, cmap="hot", vmin=0, vmax=1)
                    axes[mi * 2 + 1, col].axis("off")

                    if j == 0:
                        axes[mi * 2, col].set_ylabel(mode, fontsize=8)

                if j == 0 and mi == 0:
                    axes[0, ti * n_per_cell].set_title(f"turb={turb}", fontsize=8)

                avg_mse = float(np.mean(mses))
                avg_ssim = float(np.mean(ssims))
                rows.append({"mode": mode, "turb": turb, "mse": avg_mse, "ssim": avg_ssim})
                print(f"  mode={mode:6s}  turb={turb}  MSE={avg_mse:.5f}  SSIM={avg_ssim:.4f}")

    plt.suptitle("VAE Reconstruction Quality (128px) — row pairs: real / recon", fontsize=10)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "reconstruction_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {grid_path}")

    # --- CSV ---
    csv_path = os.path.join(output_dir, "quality_table.csv")
    with open(csv_path, "w") as f:
        f.write("mode,turb_level,mse,ssim\n")
        for r in rows:
            f.write(f"{r['mode']},{r['turb']},{r['mse']:.6f},{r['ssim']:.4f}\n")
    print(f"Saved: {csv_path}")

    # --- Summary bar chart ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    mse_vals = [r["mse"] for r in rows]
    ssim_vals = [r["ssim"] for r in rows]
    labels = [f"{r['mode']}\nt={r['turb']}" for r in rows]

    ax1.bar(range(len(rows)), mse_vals, color="steelblue")
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_ylabel("MSE")
    ax1.set_title("Reconstruction MSE per (mode, turb)")

    ax2.bar(range(len(rows)), ssim_vals, color="darkorange")
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(labels, fontsize=6)
    ax2.set_ylabel("SSIM")
    ax2.set_title("Reconstruction SSIM per (mode, turb)")

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "quality_barchart.png")
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--output_dir", default="analysis_vae_quality")
    parser.add_argument("--n_per_cell", type=int, default=8,
                        help="Samples per (mode, turb) cell")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(**vars(args))
