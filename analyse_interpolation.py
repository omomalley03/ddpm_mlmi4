"""
Track C2: Latent Interpolation — Ho et al. Section 4.3 style.

Mirrors the CelebA face interpolation from the original DDPM paper.

Method (per noise level t*):
  1. Take one real Gaussian beam x₀_a and one real OAM p4 beam x₀_b.
  2. Sample a shared noise vector ε ~ N(0, I).
  3. Noise both to timestep t*:
       x_{t*,a} = q_sample(x₀_a, t*, ε)
       x_{t*,b} = q_sample(x₀_b, t*, ε)
     Using the same ε isolates the image content from the noise realization.
  4. For α ∈ {0.0, 0.125, …, 1.0}: slerp between x_{t*,a} and x_{t*,b}.
  5. Reverse-diffuse each slerped point from t* down to 0.

Output grid:
  - Rows: noise levels t* ∈ {250, 500, 750, 999}
  - Columns: interpolation steps α = 0 → 1
  - Top header row: original images + their noised versions

Usage:
    python analyse_interpolation.py \
        --checkpoint checkpoints_ddpm_oam/ckpt_300000.pt \
        --mat_path croped_2_2_pupil_data.mat \
        --output_dir analysis_interpolation \
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


# ── Interpolation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def interpolation_grid(
    model,
    diffusion,
    dataset,
    device,
    output_dir,
    t_stars=(250, 500, 750, 999),
    n_steps=9,
    mode_a=0,   # gauss
    mode_b=1,   # p4
    n_source_pairs=3,   # how many (x₀_a, x₀_b) pairs to visualise per t*
    turb_level=None,    # None = lowest; int = specific level; -1 = all levels
):
    """Generate the Ho et al.-style slerp interpolation grid.

    For each noise level t*:
      - Noise x₀_a and x₀_b with the SAME ε vector.
      - Slerp between the two noised images at n_steps α values.
      - Reverse diffuse from t* to 0.

    Grid rows: t* values
    Grid columns: α = 0, …, 1
    Additional header: the two original source images (endpoints).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve which turbulence levels to iterate over
    if turb_level == -1:
        turb_levels_to_run = dataset.turb_categories
    elif turb_level is None:
        turb_levels_to_run = [min(dataset.turb_categories)]
    else:
        if turb_level not in dataset.turb_categories:
            raise ValueError(
                f"turb_level={turb_level} not in dataset. "
                f"Available: {dataset.turb_categories}"
            )
        turb_levels_to_run = [turb_level]

    alphas = np.linspace(0.0, 1.0, n_steps)

    for tau in turb_levels_to_run:
        print(f"\nTurbulence level: {tau}")

        idx_a = np.where(
            (dataset.mode_labels == mode_a) & (dataset.turb_labels == tau)
        )[0]
        idx_b = np.where(
            (dataset.mode_labels == mode_b) & (dataset.turb_labels == tau)
        )[0]

        if len(idx_a) == 0 or len(idx_b) == 0:
            print(f"  No images found for turb {tau} in one or both modes — skipping.")
            continue

        n_pairs = min(n_source_pairs, len(idx_a), len(idx_b))
        rng = np.random.RandomState(0)
        sel_a = rng.choice(idx_a, n_pairs, replace=False)
        sel_b = rng.choice(idx_b, n_pairs, replace=False)

        for pair_idx, (ia, ib) in enumerate(zip(sel_a, sel_b)):
            x0_a = dataset[int(ia)][0].unsqueeze(0).to(device)   # (1,1,H,W)
            x0_b = dataset[int(ib)][0].unsqueeze(0).to(device)

            # ── Build figure ───────────────────────────────────────────────
            # Rows: t* levels; Columns: α interpolation steps
            # Extra first row: original endpoints
            n_rows = len(t_stars) + 1
            n_cols = n_steps

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 1.9, n_rows * 2.0))

            # ── Header row: original images ────────────────────────────────
            for col in range(n_cols):
                axes[0, col].axis("off")

            # Show x₀_a at left, x₀_b at right, blank in between
            _show(axes[0, 0], x0_a.squeeze().cpu().numpy(),
                  f"{dataset.mode_display_name(mode_a)}\n(source)")
            _show(axes[0, -1], x0_b.squeeze().cpu().numpy(),
                  f"{dataset.mode_display_name(mode_b)}\n(source)")

            # ── Interpolation rows ─────────────────────────────────────────
            for row, t_star in enumerate(t_stars):
                t_int = min(int(t_star), diffusion.T - 1)

                # Fixed shared noise (same ε for both images → controlled interpolation)
                eps = torch.randn_like(x0_a)

                t_tensor_a = torch.full((1,), t_int, dtype=torch.long, device=device)
                x_t_a = diffusion.q_sample(x0_a, t_tensor_a, noise=eps)
                x_t_b = diffusion.q_sample(x0_b, t_tensor_a, noise=eps)

                for col, alpha in enumerate(alphas):
                    # Slerp between the two noised images
                    z_interp = diffusion.slerp(x_t_a, x_t_b, alpha)

                    # Reverse diffuse from t* to 0
                    x_rec = diffusion.p_sample_loop_from_t(model, z_interp, t_int)

                    _show(axes[row + 1, col], x_rec.squeeze().cpu().numpy(),
                          f"α={alpha:.2f}" if row == len(t_stars) - 1 else "")

                # Row label
                axes[row + 1, 0].set_ylabel(f"t*={t_star}", fontsize=8)

            # ── Axes labels ────────────────────────────────────────────────
            axes[0, 0].set_ylabel("Source images", fontsize=8)
            for col, alpha in enumerate(alphas):
                axes[1, col].set_title(f"α={alpha:.2f}", fontsize=7)

            plt.suptitle(
                f"OAM Beam Interpolation (Ho et al. style) — turb {tau}, pair {pair_idx+1}\n"
                f"{dataset.mode_display_name(mode_a)} → {dataset.mode_display_name(mode_b)}  |  "
                f"Rows: noise level t*  |  Columns: slerp α",
                fontsize=9,
            )
            plt.tight_layout()

            fname = f"interpolation_turb{tau}_pair{pair_idx+1}.png"
            path = os.path.join(output_dir, fname)
            plt.savefig(path, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"Saved: {path}")


@torch.no_grad()
def interpolation_grid_turb(
    model,
    diffusion,
    dataset,
    device,
    output_dir,
    t_stars=(250, 500, 750, 999),
    n_steps=9,
    mode_idx=0,      # fixed mode for both endpoints
    turb_a=None,     # turbulence label of the "clean" endpoint
    turb_b=None,     # turbulence label of the "turbulent" endpoint
    n_source_pairs=3,
):
    """Ho et al.-style slerp grid, interpolating between turbulence levels.

    Mode is fixed; the two endpoints differ only in turbulence strength.
    E.g. Gaussian at turb=1 (left) → Gaussian at turb=3 (right).

    Rows: noise level t*  |  Columns: slerp α = 0 → 1
    Header row: the two source images (turb_a on left, turb_b on right).
    """
    os.makedirs(output_dir, exist_ok=True)

    if turb_a not in dataset.turb_categories:
        raise ValueError(f"turb_a={turb_a} not in dataset. Available: {dataset.turb_categories}")
    if turb_b not in dataset.turb_categories:
        raise ValueError(f"turb_b={turb_b} not in dataset. Available: {dataset.turb_categories}")

    idx_a = np.where(
        (dataset.mode_labels == mode_idx) & (dataset.turb_labels == turb_a)
    )[0]
    idx_b = np.where(
        (dataset.mode_labels == mode_idx) & (dataset.turb_labels == turb_b)
    )[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(
            f"No images found for mode={mode_idx} at turb_a={turb_a} or turb_b={turb_b}."
        )

    n_pairs = min(n_source_pairs, len(idx_a), len(idx_b))
    rng = np.random.RandomState(0)
    sel_a = rng.choice(idx_a, n_pairs, replace=False)
    sel_b = rng.choice(idx_b, n_pairs, replace=False)
    alphas = np.linspace(0.0, 1.0, n_steps)
    mode_name = dataset.mode_display_name(mode_idx)

    for pair_idx, (ia, ib) in enumerate(zip(sel_a, sel_b)):
        x0_a = dataset[int(ia)][0].unsqueeze(0).to(device)
        x0_b = dataset[int(ib)][0].unsqueeze(0).to(device)

        n_rows = len(t_stars) + 1
        n_cols = n_steps
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 1.9, n_rows * 2.0))

        # Header row: source images
        for col in range(n_cols):
            axes[0, col].axis("off")
        _show(axes[0, 0], x0_a.squeeze().cpu().numpy(),
              f"{mode_name}\nturb={turb_a}")
        _show(axes[0, -1], x0_b.squeeze().cpu().numpy(),
              f"{mode_name}\nturb={turb_b}")

        # Interpolation rows
        for row, t_star in enumerate(t_stars):
            t_int = min(int(t_star), diffusion.T - 1)
            eps = torch.randn_like(x0_a)   # same noise for both endpoints

            t_tensor = torch.full((1,), t_int, dtype=torch.long, device=device)
            x_t_a = diffusion.q_sample(x0_a, t_tensor, noise=eps)
            x_t_b = diffusion.q_sample(x0_b, t_tensor, noise=eps)

            for col, alpha in enumerate(alphas):
                z_interp = diffusion.slerp(x_t_a, x_t_b, alpha)
                x_rec = diffusion.p_sample_loop_from_t(model, z_interp, t_int)
                _show(axes[row + 1, col], x_rec.squeeze().cpu().numpy(),
                      f"α={alpha:.2f}" if row == len(t_stars) - 1 else "")

            axes[row + 1, 0].set_ylabel(f"t*={t_star}", fontsize=8)

        axes[0, 0].set_ylabel("Source images", fontsize=8)
        for col, alpha in enumerate(alphas):
            axes[1, col].set_title(f"α={alpha:.2f}", fontsize=7)

        plt.suptitle(
            f"OAM Turbulence Interpolation (Ho et al. style) — {mode_name}, pair {pair_idx+1}\n"
            f"turb={turb_a} → turb={turb_b}  |  Rows: noise level t*  |  Columns: slerp α",
            fontsize=9,
        )
        plt.tight_layout()

        fname = f"interpolation_{dataset.modes[mode_idx]}_turb{turb_a}_to_turb{turb_b}_pair{pair_idx+1}.png"
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def _show(ax, img_np, title=""):
    img_np = ((img_np + 1) / 2).clip(0, 1)
    ax.imshow(img_np, cmap="hot")
    if title:
        ax.set_title(title, fontsize=7)
    ax.axis("off")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained DDPM checkpoint (EMA weights)")
    parser.add_argument("--mat_path", required=True,
                        help="Path to OAM .mat data file")
    parser.add_argument("--output_dir", default="analysis_interpolation")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_steps", type=int, default=9,
                        help="Number of interpolation α steps (including endpoints)")
    parser.add_argument("--n_pairs", type=int, default=3,
                        help="Number of (source_a, source_b) pairs to visualise")
    parser.add_argument(
        "--t_stars", type=int, nargs="+", default=[250, 500, 750, 999],
        help="Noise levels at which to perform interpolation"
    )
    parser.add_argument(
        "--turb_level", type=int, default=-1,
        help="Turbulence level to use for source images. "
             "-1 = all levels (one grid per level); "
             "0,1,2,... = specific level"
    )
    parser.add_argument("--mode_a", type=int, default=0,
                        help="Dataset index of the first (left) mode. Default: 0 (gauss).")
    parser.add_argument("--mode_b", type=int, default=1,
                        help="Dataset index of the second (right) mode. Default: 1.")
    parser.add_argument("--turb_a", type=int, default=None,
                        help="Turbulence interpolation: label of the low-turb endpoint. "
                             "When set alongside --turb_b, switches to turbulence interpolation "
                             "(fixed mode=--mode_a, varying turb_a→turb_b).")
    parser.add_argument("--turb_b", type=int, default=None,
                        help="Turbulence interpolation: label of the high-turb endpoint.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diffusion = GaussianDiffusion(T=1000, device=device)
    model = load_model(args.checkpoint, args.image_size, device)
    dataset = OAMDataset(args.mat_path, image_size=args.image_size)

    print(f"Dataset: {len(dataset)} images | Modes: {dataset.modes}")
    print(f"t* values: {args.t_stars} | α steps: {args.n_steps} | pairs: {args.n_pairs}")

    turb_interp_mode = args.turb_a is not None and args.turb_b is not None

    if turb_interp_mode:
        print(f"\n--- Turbulence interpolation: mode={args.mode_a} "
              f"turb {args.turb_a} → {args.turb_b} ---")
        interpolation_grid_turb(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            device=device,
            output_dir=args.output_dir,
            t_stars=args.t_stars,
            n_steps=args.n_steps,
            mode_idx=args.mode_a,
            turb_a=args.turb_a,
            turb_b=args.turb_b,
            n_source_pairs=args.n_pairs,
        )
    else:
        if len(dataset.modes) < 2:
            raise ValueError(
                "Need at least 2 OAM modes for mode interpolation. "
                "Check MODES in dataset_oam.py — should include 'gauss' and at least one OAM mode."
            )
        print("\n--- Mode interpolation ---")
        interpolation_grid(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            device=device,
            output_dir=args.output_dir,
            t_stars=args.t_stars,
            n_steps=args.n_steps,
            mode_a=args.mode_a,
            mode_b=args.mode_b,
            n_source_pairs=args.n_pairs,
            turb_level=args.turb_level,
        )

    print(f"\nDone. Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
