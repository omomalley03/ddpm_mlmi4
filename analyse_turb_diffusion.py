"""
Track B: Turbulence as Diffusion Analogy.

For each turbulence level τ, find the "equivalent diffusion timestep" t*(τ)
by minimising E[||q_sample(x_clean, t) - x_turb||²] over t.

If the mapping t*(τ) is monotonically increasing, it suggests that atmospheric
turbulence corrupts OAM beams in a way that is statistically analogous to the
DDPM forward diffusion process.

Outputs:
  mse_vs_t.png        — MSE curves per turbulence level with t* markers
  t_star_mapping.png  — turbulence label vs t* scatter
  img2img_demo.png    — turbulent images "denoised" by the DDPM reverse process

Usage:
    python analyse_turb_diffusion.py \
        --checkpoint checkpoints_ddpm_oam/ckpt_300000.pt \
        --mat_path croped_2_2_pupil_data.mat \
        --output_dir analysis_turb \
        --image_size 128
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import GaussianDiffusion
from model import UNet
from dataset_oam import OAMDataset

OAM_CHANNEL_MULTS = (1, 2, 4, 4, 4)
OAM_BASE_CHANNELS = 64


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, image_size, device):
    model = UNet(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=image_size,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded EMA checkpoint from step {ckpt['step']}")
    return model


# ── Track B core: MSE curves ──────────────────────────────────────────────────

def compute_mse_curves(dataset, diffusion, t_values, n_pairs=200, device="cpu"):
    """For each turbulence level τ, compute E[||q(x_clean, t) - x_turb||²] vs t.

    We compare within-mode pairs only (gauss→gauss, p4→p4) and average.
    x_clean = images at the lowest turbulence level.

    Returns:
        mse_curves: dict {turb_label: np.array shape (len(t_values),)}
        t_stars:    list of (turb_label, t*) tuples
    """
    turb_cats = dataset.turb_categories
    min_turb = turb_cats[0]
    n_modes = len(dataset.modes)

    mse_curves = {tau: np.zeros(len(t_values)) for tau in turb_cats}

    for tau in turb_cats:
        print(f"  Computing MSE for turb level {tau}...")
        mode_mse = np.zeros(len(t_values))

        for mode_idx in range(n_modes):
            clean_idx = np.where(
                (dataset.turb_labels == min_turb) & (dataset.mode_labels == mode_idx)
            )[0]
            turb_idx = np.where(
                (dataset.turb_labels == tau) & (dataset.mode_labels == mode_idx)
            )[0]

            if len(clean_idx) == 0 or len(turb_idx) == 0:
                continue

            rng = np.random.RandomState(42 + mode_idx)
            n = min(n_pairs, len(clean_idx), len(turb_idx))
            c_sel = rng.choice(clean_idx, n, replace=False)
            t_sel = rng.choice(turb_idx, n, replace=False)

            for i, t in enumerate(t_values):
                t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
                batch_size = 32
                mse_sum = 0.0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    c_imgs = torch.stack(
                        [dataset[int(j)][0] for j in c_sel[start:end]]
                    ).to(device)
                    turb_imgs = torch.stack(
                        [dataset[int(j)][0] for j in t_sel[start:end]]
                    ).to(device)

                    t_batch = t_tensor.expand(len(c_imgs))
                    # Average over multiple noise realizations for stability
                    mse_per_img = 0.0
                    for _ in range(3):
                        x_noised = diffusion.q_sample(c_imgs, t_batch)
                        mse_per_img += ((x_noised - turb_imgs) ** 2).mean(dim=(1, 2, 3))
                    mse_per_img /= 3
                    mse_sum += mse_per_img.sum().item()

                mode_mse[i] += mse_sum / n

        mse_curves[tau] = mode_mse / n_modes
        t_star = t_values[int(np.argmin(mse_curves[tau]))]
        print(f"    t* = {t_star}")

    t_stars = [(tau, t_values[int(np.argmin(mse_curves[tau]))]) for tau in turb_cats]
    return mse_curves, t_stars


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_mse_curves(mse_curves, t_values, t_stars, output_dir):
    """MSE vs diffusion timestep, one curve per turbulence level."""
    t_star_map = dict(t_stars)
    turb_levels = sorted(mse_curves.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(turb_levels)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for tau, color in zip(turb_levels, colors):
        mse = mse_curves[tau]
        t_star = t_star_map[tau]
        ax.plot(t_values, mse, color=color, label=f"Turb {tau}  (t*={t_star})",
                linewidth=1.8)
        ax.axvline(t_star, color=color, linestyle="--", alpha=0.35, linewidth=0.9)

    ax.set_xlabel("Diffusion timestep  t", fontsize=11)
    ax.set_ylabel(r"MSE :  $\| q(x_\mathrm{clean},\,t) - x_\mathrm{turb}\|^2$", fontsize=10)
    ax.set_title("Finding the DDPM-equivalent noise level for each turbulence strength", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "mse_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_t_star_mapping(t_stars, output_dir):
    """Scatter: turbulence label vs t*."""
    taus = [x[0] for x in t_stars]
    stars = [x[1] for x in t_stars]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, stars, "o-", color="steelblue", linewidth=2, markersize=9,
            markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Turbulence strength  τ", fontsize=11)
    ax.set_ylabel("Equivalent diffusion timestep  t*", fontsize=11)
    ax.set_title("Turbulence Strength → DDPM Timestep Mapping", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "t_star_mapping.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_img2img_demo(model, diffusion, dataset, t_stars, device, output_dir,
                      extra_noise_steps=(0, 100, 250)):
    """Img2img demo: noise turbulent image to t*, reverse with DDPM.

    Grid layout:
        Columns: [clean ref] [turbulent input] [reversed from t*] [reversed from t*+100] ...
        Rows: one per turbulence level (using gauss mode for clarity).
    """
    turb_cats = sorted(dataset.turb_categories)
    t_star_map = dict(t_stars)
    min_turb = turb_cats[0]
    mode_idx = 0  # gauss

    n_cols = 2 + len(extra_noise_steps)
    fig, axes = plt.subplots(len(turb_cats), n_cols,
                             figsize=(n_cols * 2.2, len(turb_cats) * 2.4))
    if len(turb_cats) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, tau in enumerate(turb_cats):
            t_star = min(t_star_map[tau], diffusion.T - 1)

            clean_indices = np.where(
                (dataset.turb_labels == min_turb) & (dataset.mode_labels == mode_idx)
            )[0]
            turb_indices = np.where(
                (dataset.turb_labels == tau) & (dataset.mode_labels == mode_idx)
            )[0]

            if len(clean_indices) == 0 or len(turb_indices) == 0:
                continue

            x_clean = dataset[int(clean_indices[0])][0].unsqueeze(0).to(device)
            x_turb = dataset[int(turb_indices[0])][0].unsqueeze(0).to(device)

            _show(axes[row, 0], x_clean.squeeze().cpu().numpy(),
                  f"Clean\n(turb {min_turb})")
            _show(axes[row, 1], x_turb.squeeze().cpu().numpy(),
                  f"Turbulent\n(turb {tau})")

            for col_offset, extra in enumerate(extra_noise_steps):
                t_img2img = min(t_star + extra, diffusion.T - 1)
                t_tensor = torch.full((1,), t_img2img, dtype=torch.long, device=device)
                x_t = diffusion.q_sample(x_turb, t_tensor)
                x_recovered = diffusion.p_sample_loop_from_t(model, x_t, t_img2img)
                label = f"t*+{extra}\n→ DDPM" if extra > 0 else f"t*={t_star}\n→ DDPM"
                _show(axes[row, 2 + col_offset],
                      x_recovered.squeeze().cpu().numpy(), label)

            axes[row, 0].set_ylabel(f"turb {tau}", fontsize=8)

    plt.suptitle("OAM Turbulence 'Removal' via DDPM Reverse Process", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "img2img_demo.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _show(ax, img_np, title):
    img_np = ((img_np + 1) / 2).clip(0, 1)
    ax.imshow(img_np, cmap="hot")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained DDPM checkpoint")
    parser.add_argument("--mat_path", required=True,
                        help="Path to OAM .mat data file")
    parser.add_argument("--output_dir", default="analysis_turb")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_pairs", type=int, default=200,
                        help="Image pairs per turbulence level for MSE estimation")
    parser.add_argument("--t_step", type=int, default=50,
                        help="Step size between t values evaluated (e.g. 50 → 0,50,100,...,950)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diffusion = GaussianDiffusion(T=1000, device=device)
    model = load_model(args.checkpoint, args.image_size, device)
    dataset = OAMDataset(args.mat_path, image_size=args.image_size)
    print(f"Dataset: {len(dataset)} images | "
          f"Modes: {dataset.modes} | "
          f"Turb categories: {dataset.turb_categories}")

    t_values = list(range(0, 1000, args.t_step))

    print("\n--- Computing MSE curves (this may take a few minutes) ---")
    mse_curves, t_stars = compute_mse_curves(
        dataset, diffusion, t_values, n_pairs=args.n_pairs, device=device
    )

    print("\n--- Plotting MSE curves ---")
    plot_mse_curves(mse_curves, t_values, t_stars, args.output_dir)

    print("\n--- Plotting t* mapping ---")
    plot_t_star_mapping(t_stars, args.output_dir)

    print("\n--- Running img2img demo ---")
    plot_img2img_demo(model, diffusion, dataset, t_stars, device, args.output_dir)

    print(f"\nDone. Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
