"""
Latent space analysis for the OAM VAE.

Two visualisations:

1. PCA traversal
   - Encode all images → flatten latent means to (N, C*H*W)
   - Run PCA, find top-K principal components
   - For each PC: start from the dataset mean latent, walk along the PC direction
     at evenly-spaced steps from -n_sigma to +n_sigma standard deviations
   - Decode each step → grid of rows (PCs) × columns (steps)
   - Tells you what each latent direction controls (mode? turbulence? intensity?)

2. t-SNE scatter
   - Project flattened latent means to 2D with t-SNE
   - Plot coloured by mode, with marker shape encoding turbulence level
   - Shows whether the latent space has learned separate clusters per mode/turb

Usage:
    python analyse_latent_space.py \
        --vae_checkpoint checkpoints_vae_128_modelA/vae_oam_epoch100.pt \
        --mat_path croped_2_2_pupil_data.mat \
        --output_dir analysis_latent_space_modelA \
        --modes gauss \
        --turb_levels 1 2 3
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from vae import VAE
from dataset_oam import OAMDataset

# VAE config — must match retrained 128px VAE
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM = 4
VAE_BASE_CHANNELS = 64
VAE_IMAGE_SIZE = 128
LATENT_SIZE = VAE_IMAGE_SIZE // (2 ** len(VAE_CHANNEL_MULTS))  # 8
LATENT_FLAT = VAE_LATENT_DIM * LATENT_SIZE * LATENT_SIZE        # 256


def load_vae(checkpoint_path, device):
    vae = VAE(
        in_channels=1,
        base_channels=VAE_BASE_CHANNELS,
        channel_mults=VAE_CHANNEL_MULTS,
        latent_dim=VAE_LATENT_DIM,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE from epoch {ckpt['epoch']}")
    return vae


@torch.no_grad()
def encode_all(vae, dataset, device, batch_size=64):
    """Encode all images; return flattened latent means, mode labels, turb labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_mu, all_mode, all_turb = [], [], []
    for imgs, modes, turbs in loader:
        imgs = imgs.to(device)
        _, mu, _ = vae.encode(imgs)          # (B, C, H, W)
        all_mu.append(mu.cpu().numpy().reshape(len(imgs), -1))  # (B, C*H*W)
        all_mode.append(modes.numpy())
        all_turb.append(turbs.numpy())
    mu = np.concatenate(all_mu, axis=0)     # (N, 256)
    modes = np.concatenate(all_mode, axis=0)
    turbs = np.concatenate(all_turb, axis=0)
    print(f"Encoded {len(mu)} images → latent shape per image: ({LATENT_FLAT},)")
    return mu, modes, turbs


# ── PCA traversal ─────────────────────────────────────────────────────────────

@torch.no_grad()
def pca_traversal(vae, mu_all, device, output_dir,
                  n_components=6, n_steps=9, n_sigma=2.5):
    """
    For each of the top n_components PCs:
      - Start from the mean latent
      - Walk from -n_sigma to +n_sigma standard deviations along the PC
      - Decode each step
    Grid: rows = PCs, columns = steps from -n_sigma to +n_sigma
    """
    pca = PCA(n_components=n_components)
    pca.fit(mu_all)
    var_ratio = pca.explained_variance_ratio_

    mean_latent = mu_all.mean(axis=0)                    # (256,)
    # Std dev of scores along each PC (= sqrt of eigenvalue)
    stds = np.sqrt(pca.explained_variance_)              # (n_components,)

    steps = np.linspace(-n_sigma, n_sigma, n_steps)      # e.g. [-2.5, ..., +2.5]
    col_labels = [f"{s:+.1f}σ" for s in steps]

    fig, axes = plt.subplots(n_components, n_steps,
                             figsize=(n_steps * 1.5, n_components * 1.8))

    for pc_idx in range(n_components):
        direction = pca.components_[pc_idx]              # (256,)
        std = stds[pc_idx]

        for col, scale in enumerate(steps):
            z_flat = mean_latent + scale * std * direction
            z = torch.tensor(
                z_flat.reshape(1, VAE_LATENT_DIM, LATENT_SIZE, LATENT_SIZE),
                dtype=torch.float32, device=device,
            )
            img = vae.decode(z).squeeze().cpu().numpy()
            vis = ((img + 1) / 2).clip(0, 1)

            axes[pc_idx, col].imshow(vis, cmap="hot", vmin=0, vmax=1)
            axes[pc_idx, col].axis("off")

        # Row label: PC index + variance explained
        axes[pc_idx, 0].set_ylabel(
            f"PC{pc_idx+1}\n({var_ratio[pc_idx]*100:.1f}% var)",
            fontsize=8,
        )

    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=7)

    plt.suptitle(
        "PCA traversal of VAE latent space\n"
        "Rows: principal components (sorted by variance explained) | "
        "Columns: ± standard deviations from dataset mean",
        fontsize=9,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "latent_pca_traversal.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also save a bar chart of variance explained
    fig2, ax = plt.subplots(figsize=(max(4, n_components * 0.8), 3))
    ax.bar(range(1, n_components + 1), var_ratio * 100, color="steelblue")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance explained — VAE latent space")
    ax.set_xticks(range(1, n_components + 1))
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, "latent_pca_variance.png")
    plt.savefig(out_path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


# ── t-SNE scatter ─────────────────────────────────────────────────────────────

def tsne_scatter(mu_all, mode_labels, turb_labels, dataset, output_dir,
                 perplexity=30, max_points=2000):
    """
    Project latent means to 2D with t-SNE.
    Colour = OAM mode, marker shape = turbulence level.
    """
    # Subsample if dataset is large (t-SNE is O(N^2))
    N = len(mu_all)
    if N > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_points, replace=False)
        mu_sub = mu_all[idx]
        modes_sub = mode_labels[idx]
        turbs_sub = turb_labels[idx]
        print(f"t-SNE: subsampled {max_points}/{N} points")
    else:
        mu_sub, modes_sub, turbs_sub = mu_all, mode_labels, turb_labels

    print(f"Running t-SNE (perplexity={perplexity})...")
    proj = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(mu_sub)

    unique_modes = np.unique(modes_sub)
    unique_turbs = sorted(np.unique(turbs_sub))
    mode_names = dataset.modes

    # Colour palette for modes
    cmap = plt.cm.get_cmap("tab10", len(unique_modes))
    mode_colors = {m: cmap(i) for i, m in enumerate(unique_modes)}

    # Marker shapes for turb levels (cycle through a fixed list)
    marker_list = ["o", "s", "^", "D", "v", "P", "*"]
    turb_markers = {t: marker_list[i % len(marker_list)]
                    for i, t in enumerate(unique_turbs)}

    fig, ax = plt.subplots(figsize=(8, 7))

    for mode_idx in unique_modes:
        for turb in unique_turbs:
            mask = (modes_sub == mode_idx) & (turbs_sub == turb)
            if not mask.any():
                continue
            name = mode_names[mode_idx] if mode_idx < len(mode_names) else str(mode_idx)
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=[mode_colors[mode_idx]],
                marker=turb_markers[turb],
                s=25, alpha=0.65, linewidths=0,
                label=f"{name}, turb={turb}",
            )

    # Legend: de-duplicate, keep tidy
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, markerscale=1.4,
              loc="best", framealpha=0.8)
    ax.set_title(
        "t-SNE of VAE latent means\n"
        "Colour = OAM mode  |  Marker shape = turbulence level",
        fontsize=10,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "latent_tsne.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--output_dir", default="analysis_latent_space")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="Modes to load, e.g. --modes gauss. "
                             "None = all modes. Model A: gauss. Model B: gauss p1 p2 p3 p4.")
    parser.add_argument("--turb_levels", type=int, nargs="+", default=None,
                        help="Turb levels to load, e.g. --turb_levels 1 2 3. "
                             "None = all levels. Model A: 1 2 3. Model B: 3.")
    parser.add_argument("--n_pca", type=int, default=6,
                        help="Number of PCA components to traverse (rows in traversal grid)")
    parser.add_argument("--n_steps", type=int, default=9,
                        help="Number of decode steps per PC (-n_sigma to +n_sigma)")
    parser.add_argument("--n_sigma", type=float, default=2.5,
                        help="Range of traversal in standard deviations")
    parser.add_argument("--tsne_perplexity", type=float, default=30)
    parser.add_argument("--tsne_max_points", type=int, default=2000,
                        help="Subsample to this many points before running t-SNE")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae = load_vae(args.vae_checkpoint, device)
    dataset = OAMDataset(args.mat_path, image_size=VAE_IMAGE_SIZE,
                         modes=args.modes, turb_levels=args.turb_levels)
    print(f"Dataset: {len(dataset)} images | modes={dataset.modes} | "
          f"turb categories={dataset.turb_categories}")

    mu_all, mode_labels, turb_labels = encode_all(vae, dataset, device)

    print("\n--- PCA traversal ---")
    pca_traversal(vae, mu_all, device, args.output_dir,
                  n_components=args.n_pca,
                  n_steps=args.n_steps,
                  n_sigma=args.n_sigma)

    print("\n--- t-SNE scatter ---")
    tsne_scatter(mu_all, mode_labels, turb_labels, dataset, args.output_dir,
                 perplexity=args.tsne_perplexity,
                 max_points=args.tsne_max_points)

    print(f"\nDone. Outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()
