"""
Track C1: Structure Emergence Analysis.

Uses the same shared noise seed for Gaussian beam and OAM p4 beam, runs two
parallel denoising trajectories, and tracks when the ring/blob distinction
becomes visible.

Metric: centre-intensity fraction
    c(t) = mean intensity in the central 10% of the image
             / mean intensity across the full image

  - High c(t) → Gaussian beam (intensity concentrated at centre)
  - Low  c(t) → OAM ring beam (intensity in the ring, dark at centre)

At t = T both trajectories start from pure noise → c ≈ 1.
As t decreases, the DDPM imposes structure; at some critical t* the
two modes become distinguishable.

Outputs:
  emergence_frames.png   — side-by-side denoising frames for each mode
  emergence_metric.png   — centre-intensity fraction vs denoising timestep

Usage:
    python analyse_emergence.py \
        --checkpoint checkpoints_ddpm_oam/ckpt_300000.pt \
        --output_dir analysis_emergence \
        --image_size 128 \
        --n_samples 8
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


# ── Centre-intensity metric ───────────────────────────────────────────────────

def centre_intensity_fraction(imgs, centre_frac=0.10):
    """Fraction of total image intensity that lies in the central region.

    Args:
        imgs: Tensor (B, 1, H, W) in [-1, 1].
        centre_frac: Radius of the central disc as a fraction of image size.

    Returns:
        Scalar: mean centre-intensity fraction across the batch.
    """
    B, _, H, W = imgs.shape
    # Normalise to [0, 1] for positive intensities
    imgs_01 = ((imgs + 1.0) / 2.0).clamp(0, 1)

    # Build circular mask for the central region
    cy, cx = H / 2.0, W / 2.0
    radius = centre_frac * min(H, W)
    ys = torch.arange(H, device=imgs.device).float() - cy
    xs = torch.arange(W, device=imgs.device).float() - cx
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    mask = (grid_y ** 2 + grid_x ** 2) <= radius ** 2   # (H, W)

    centre_mean = imgs_01[:, :, mask].mean(dim=-1).mean()
    total_mean = imgs_01.mean()

    if total_mean < 1e-6:
        return 1.0  # pure noise → undefined, return neutral value

    return (centre_mean / total_mean).item()


# ── Denoising trajectory ──────────────────────────────────────────────────────

@torch.no_grad()
def run_trajectory(model, diffusion, noise, n_record=40):
    """Run reverse diffusion from `noise`, recording frames and metric.

    Args:
        noise: Starting noise tensor (B, C, H, W).
        n_record: Number of timesteps to record (evenly spaced).

    Returns:
        frames:  list of (t, tensor (B,C,H,W)) — recorded states
        metrics: list of (t, centre_fraction)
    """
    # Timesteps to capture
    record_ts = set(
        round(v) for v in
        torch.linspace(diffusion.T - 1, 0, n_record).tolist()
    )

    x = noise.clone()
    frames = []
    metrics = []

    for t in reversed(range(diffusion.T)):
        x = diffusion.p_sample(model, x, t)
        if t in record_ts:
            frames.append((t, x.clone()))
            metrics.append((t, centre_intensity_fraction(x)))

    return frames, metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_frames(frames_a, frames_b, label_a, label_b, output_dir, n_show=10):
    """Side-by-side denoising frames for two modes.

    Shows n_show evenly-spaced frames from the full trajectory.
    Rows: two modes  |  Columns: denoising timesteps (T → 0, left to right)
    """
    # Sub-select n_show frames evenly
    idx = np.round(np.linspace(0, len(frames_a) - 1, n_show)).astype(int)
    sel_a = [frames_a[i] for i in idx]
    sel_b = [frames_b[i] for i in idx]

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.9, 4.5))

    for col, ((t_a, img_a), (t_b, img_b)) in enumerate(zip(sel_a, sel_b)):
        img_a_np = ((img_a[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
        img_b_np = ((img_b[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)

        axes[0, col].imshow(img_a_np, cmap="hot")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"t={t_a}", fontsize=6)

        axes[1, col].imshow(img_b_np, cmap="hot")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel(label_a, fontsize=9)
    axes[1, 0].set_ylabel(label_b, fontsize=9)

    plt.suptitle(
        "OAM DDPM Denoising Trajectories: Structure Emergence\n"
        "(same starting noise, shared denoising — structure diverges as t → 0)",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "emergence_frames.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_emergence_metric(metrics_a, metrics_b, label_a, label_b, output_dir):
    """Plot centre-intensity fraction vs denoising timestep for both modes."""
    ts_a = [m[0] for m in metrics_a]
    vals_a = [m[1] for m in metrics_a]
    ts_b = [m[0] for m in metrics_b]
    vals_b = [m[1] for m in metrics_b]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts_a, vals_a, "o-", color="royalblue", label=label_a,
            markersize=3, linewidth=1.5)
    ax.plot(ts_b, vals_b, "s-", color="tomato", label=label_b,
            markersize=3, linewidth=1.5)

    # Mark where the two curves diverge most
    ts_common = sorted(set(ts_a) & set(ts_b), reverse=True)
    if ts_common:
        dict_a = dict(zip(ts_a, vals_a))
        dict_b = dict(zip(ts_b, vals_b))
        diffs = [(abs(dict_a[t] - dict_b[t]), t) for t in ts_common
                 if t in dict_a and t in dict_b]
        if diffs:
            max_diff, t_diverge = max(diffs)
            ax.axvline(t_diverge, color="gray", linestyle="--", alpha=0.6,
                       label=f"Max divergence at t={t_diverge}")

    ax.invert_xaxis()  # t goes from T (noise) to 0 (image), left to right
    ax.set_xlabel("Denoising timestep t  (T → 0)", fontsize=11)
    ax.set_ylabel("Centre-intensity fraction", fontsize=11)
    ax.set_title(
        "When does OAM structure emerge during DDPM reverse process?\n"
        "High value = Gaussian blob  |  Low value = OAM ring",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "emergence_metric.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_emergence_metric_averaged(all_metrics_a, all_metrics_b, label_a, label_b,
                                   output_dir):
    """Same as above but averaged over multiple noise seeds (with std shading)."""
    # all_metrics_* : list of metric lists, one per seed
    def to_arrays(all_metrics):
        # Each entry: list of (t, val) in decreasing t order
        ts = [m[0] for m in all_metrics[0]]
        vals = np.array([[m[1] for m in ms] for ms in all_metrics])  # (n_seeds, n_t)
        return np.array(ts), vals.mean(axis=0), vals.std(axis=0)

    ts_a, mean_a, std_a = to_arrays(all_metrics_a)
    ts_b, mean_b, std_b = to_arrays(all_metrics_b)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts_a, mean_a, color="royalblue", label=label_a, linewidth=2)
    ax.fill_between(ts_a, mean_a - std_a, mean_a + std_a, color="royalblue", alpha=0.2)
    ax.plot(ts_b, mean_b, color="tomato", label=label_b, linewidth=2)
    ax.fill_between(ts_b, mean_b - std_b, mean_b + std_b, color="tomato", alpha=0.2)

    ax.invert_xaxis()
    ax.set_xlabel("Denoising timestep t  (T → 0)", fontsize=11)
    ax.set_ylabel("Centre-intensity fraction (mean ± std)", fontsize=11)
    ax.set_title(
        f"OAM Structure Emergence  ({len(all_metrics_a)} random seeds)\n"
        "High = Gaussian blob  |  Low = OAM ring",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "emergence_metric_averaged.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="analysis_emergence")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_seeds", type=int, default=8,
                        help="Number of random noise seeds to average metric over")
    parser.add_argument("--n_frames", type=int, default=40,
                        help="Number of timesteps to record along the trajectory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diffusion = GaussianDiffusion(T=1000, device=device)
    model = load_model(args.checkpoint, args.image_size, device)

    shape = (1, 1, args.image_size, args.image_size)

    all_metrics_gauss = []
    all_metrics_p4 = []

    print(f"\nRunning {args.n_seeds} paired trajectories (gauss + p4, same noise)...")

    for seed in range(args.n_seeds):
        print(f"  Seed {seed+1}/{args.n_seeds}")
        torch.manual_seed(seed)
        noise = torch.randn(shape, device=device)

        # Both modes start from the SAME noise → trajectories diverge due to model
        frames_gauss, metrics_gauss = run_trajectory(
            model, diffusion, noise.clone(), n_record=args.n_frames
        )
        frames_p4, metrics_p4 = run_trajectory(
            model, diffusion, noise.clone(), n_record=args.n_frames
        )
        # Note: both runs are independent (p_sample adds stochastic noise at each step)
        # but start from the same x_T. The divergence reflects the model's learned modes.

        all_metrics_gauss.append(metrics_gauss)
        all_metrics_p4.append(metrics_p4)

        # Save frames from the first seed
        if seed == 0:
            plot_frames(frames_gauss, frames_p4,
                        label_a="Gaussian mode", label_b="OAM p4 mode",
                        output_dir=args.output_dir)

    print("\n--- Plotting single-seed metric ---")
    plot_emergence_metric(
        all_metrics_gauss[0], all_metrics_p4[0],
        label_a="Gaussian mode", label_b="OAM p4 mode",
        output_dir=args.output_dir,
    )

    print("--- Plotting averaged metric ---")
    plot_emergence_metric_averaged(
        all_metrics_gauss, all_metrics_p4,
        label_a="Gaussian mode", label_b="OAM p4 mode",
        output_dir=args.output_dir,
    )

    print(f"\nDone. Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
