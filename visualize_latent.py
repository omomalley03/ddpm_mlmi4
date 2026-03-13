"""
Latent space visualization for the OAM VAE.

Four visualizations:
  1. PCA / t-SNE scatter  — project all latents to 2D, colour by mode or turbulence strength
  2. Latent interpolation — linear path between two samples in latent space
  3. PCA traversal        — walk along the top-K principal components of the latent space
  4. Reconstruction grid  — original vs reconstructed for each mode × turbulence level

Usage:
    python run.py --mode visualize_oam \
        --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt \
        --mat_path /path/to/data.mat \
        --output_dir vis_oam
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from vae import VAE
from dataset_oam import OAMDataset, MODE_DISPLAY
from train_vae_oam import OAM_CHANNEL_MULTS, OAM_LATENT_DIM, OAM_BASE_CHANNELS


# ─── Helpers ────────────────────────────────────────────────────────────────

def _load_vae(checkpoint_path, device):
    vae = VAE(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        latent_dim=OAM_LATENT_DIM,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    print(f"Loaded VAE from epoch {ckpt['epoch']}")
    return vae


@torch.no_grad()
def encode_dataset(vae, dataset, device, batch_size=64):
    """Run entire dataset through encoder. Returns flattened latent means.

    Returns:
        mus:         (N, latent_dim * H * W) float32 numpy
        mode_labels: (N,) int
        turb_labels: (N,) int
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_mus, all_modes, all_turbs = [], [], []

    for imgs, mode_labels, turb_labels in loader:
        imgs = imgs.to(device)
        _, mu, _ = vae.encode(imgs)
        all_mus.append(mu.cpu().numpy().reshape(len(imgs), -1))
        all_modes.append(mode_labels.numpy())
        all_turbs.append(turb_labels.numpy())

        if len(all_mus) % 10 == 0:
            encoded = sum(len(m) for m in all_mus)
            print(f"  Encoded {encoded}/{len(dataset)}")

    return (
        np.concatenate(all_mus, axis=0),
        np.concatenate(all_modes, axis=0),
        np.concatenate(all_turbs, axis=0),
    )


# ─── 1. PCA / t-SNE scatter ─────────────────────────────────────────────────

def plot_latent_scatter(mus, mode_labels, turb_labels, dataset, output_dir,
                        method="tsne", max_samples=5000):
    """2D scatter plot of latent space coloured by mode and turbulence strength.

    Args:
        method: "tsne" or "pca"
        max_samples: Subsample for speed if dataset is large
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    os.makedirs(output_dir, exist_ok=True)

    # Subsample if needed
    N = len(mus)
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        mus_s = mus[idx]
        mode_labels_s = mode_labels[idx]
        turb_labels_s = turb_labels[idx]
    else:
        mus_s, mode_labels_s, turb_labels_s = mus, mode_labels, turb_labels

    # Reduce dimensionality
    mus_scaled = StandardScaler().fit_transform(mus_s)

    if method == "tsne":
        print("Running PCA (50 dims) then t-SNE (2 dims)...")
        n_pca = min(50, mus_scaled.shape[1])
        coords_pca = PCA(n_components=n_pca).fit_transform(mus_scaled)
        coords = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                      init="pca", random_state=42).fit_transform(coords_pca)
        axis_labels = ["t-SNE 1", "t-SNE 2"]
    else:
        print("Running PCA (2 dims)...")
        coords = PCA(n_components=2).fit_transform(mus_scaled)
        axis_labels = ["PC 1", "PC 2"]

    n_modes = len(dataset.modes)
    mode_colors = cm.tab10(np.linspace(0, 1, n_modes))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: colour by OAM mode ──
    ax = axes[0]
    for mode_idx in range(n_modes):
        mask = mode_labels_s == mode_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[mode_colors[mode_idx]], label=dataset.mode_display_name(mode_idx),
                   alpha=0.5, s=8, linewidths=0)
    ax.set_title(f"Latent Space by OAM Mode ({method.upper()})", fontsize=12)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(markerscale=3, fontsize=8, loc="best")

    # ── Right: colour by turbulence strength ──
    ax = axes[1]
    turb_cats = sorted(np.unique(turb_labels_s).tolist())
    turb_colors = cm.plasma(np.linspace(0.1, 0.9, len(turb_cats)))
    for i, cat in enumerate(turb_cats):
        mask = turb_labels_s == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[turb_colors[i]], label=f"Turb {cat}",
                   alpha=0.5, s=8, linewidths=0)
    ax.set_title(f"Latent Space by Turbulence Strength ({method.upper()})", fontsize=12)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(markerscale=3, fontsize=8, loc="best")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"latent_{method}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── 2. Latent interpolation ─────────────────────────────────────────────────

@torch.no_grad()
def plot_interpolation(vae, dataset, device, output_dir, n_steps=10,
                       mode_a=0, mode_b=4, turb_level=None):
    """Decode a linear interpolation between two latent codes.

    Interpolates between one sample from mode_a and one from mode_b.
    Shows the full trajectory as a row of decoded images.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _get_sample(mode_idx, turb=None):
        indices = np.where(dataset.mode_labels == mode_idx)[0]
        if turb is not None:
            indices = indices[dataset.turb_labels[indices] == turb]
        idx = indices[0]
        img, _, _ = dataset[idx]
        return img.unsqueeze(0).to(device)

    img_a = _get_sample(mode_a, turb_level)
    img_b = _get_sample(mode_b, turb_level)

    _, mu_a, _ = vae.encode(img_a)
    _, mu_b, _ = vae.encode(img_b)

    alphas = np.linspace(0, 1, n_steps)
    decoded_imgs = []
    for alpha in alphas:
        z_interp = (1 - alpha) * mu_a + alpha * mu_b
        decoded = vae.decode(z_interp)
        decoded_imgs.append(decoded.squeeze().cpu().numpy())

    # Also show the original endpoints
    orig_a = img_a.squeeze().cpu().numpy()
    orig_b = img_b.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, n_steps + 2, figsize=((n_steps + 2) * 2, 2.5))

    def _show(ax, img_np, title):
        img_np = ((img_np + 1) / 2).clip(0, 1)
        ax.imshow(img_np, cmap="hot")
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    _show(axes[0], orig_a, f"{dataset.mode_display_name(mode_a)}\n(input)")
    for i, (alpha, img_np) in enumerate(zip(alphas, decoded_imgs)):
        _show(axes[i + 1], img_np, f"α={alpha:.1f}")
    _show(axes[-1], orig_b, f"{dataset.mode_display_name(mode_b)}\n(input)")

    name_a = dataset.mode_display_name(mode_a)
    name_b = dataset.mode_display_name(mode_b)
    plt.suptitle(f"Latent Interpolation: {name_a} → {name_b}", fontsize=10)
    plt.tight_layout()

    fname = f"interpolation_{dataset.modes[mode_a]}_to_{dataset.modes[mode_b]}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── 3. PCA traversal ────────────────────────────────────────────────────────

@torch.no_grad()
def plot_pca_traversal(vae, mus, dataset, device, output_dir,
                       n_components=6, n_steps=7, n_sigma=3.0):
    """Decode images while traversing the top PCA directions of the latent space.

    For each of the top-K PCA components, vary the latent code from
    -n_sigma to +n_sigma standard deviations along that direction,
    starting from the dataset mean latent.

    Output: grid where rows = PCA components, columns = traversal steps.
    """
    from sklearn.decomposition import PCA

    os.makedirs(output_dir, exist_ok=True)

    # Fit PCA on all latent means
    print(f"Fitting PCA on {len(mus)} latent codes...")
    pca = PCA(n_components=n_components)
    pca.fit(mus)

    # Latent shape: (latent_dim, H_lat, W_lat)
    sample_img, _, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    _, mu_sample, _ = vae.encode(sample_img)
    lat_shape = mu_sample.shape[1:]  # (C, H, W)
    lat_dim = mu_sample.numel() // 1  # flattened size

    # Mean latent (in flattened space)
    mean_latent = mus.mean(axis=0)  # (flat_dim,)

    alphas = np.linspace(-n_sigma, n_sigma, n_steps)
    fig, axes = plt.subplots(n_components, n_steps, figsize=(n_steps * 1.8, n_components * 2))

    for comp_idx in range(n_components):
        direction = pca.components_[comp_idx]  # (flat_dim,)
        std = np.sqrt(pca.explained_variance_[comp_idx])

        for step_idx, alpha in enumerate(alphas):
            z_flat = mean_latent + alpha * std * direction
            z = torch.from_numpy(z_flat.astype(np.float32)).to(device)
            z = z.view(1, *lat_shape)

            decoded = vae.decode(z)
            img_np = decoded.squeeze().cpu().numpy()
            img_np = ((img_np + 1) / 2).clip(0, 1)

            ax = axes[comp_idx, step_idx]
            ax.imshow(img_np, cmap="hot")
            ax.axis("off")

            if step_idx == 0:
                var_pct = pca.explained_variance_ratio_[comp_idx] * 100
                ax.set_ylabel(f"PC{comp_idx+1}\n({var_pct:.1f}%)", fontsize=8)
            if comp_idx == 0:
                ax.set_title(f"{alpha:+.1f}σ", fontsize=7)

    plt.suptitle("Latent Space PCA Traversal", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "pca_traversal.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── 4. Reconstruction grid ──────────────────────────────────────────────────

@torch.no_grad()
def plot_reconstruction_grid(vae, dataset, device, output_dir, n_per_cell=3):
    """Grid of original vs reconstructed images.

    Rows = OAM modes, columns = turbulence levels.
    Each cell shows n_per_cell original/reconstruction pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    modes = dataset.modes
    turb_cats = dataset.turb_categories
    n_modes = len(modes)
    n_turbs = len(turb_cats)

    fig, axes = plt.subplots(
        n_modes, n_turbs * n_per_cell * 2,
        figsize=(n_turbs * n_per_cell * 2.5, n_modes * 2.5)
    )
    if n_modes == 1:
        axes = axes[None, :]

    for row, mode_idx in enumerate(range(n_modes)):
        for col_t, turb_cat in enumerate(turb_cats):
            # Find samples matching this mode AND turbulence level
            indices = np.where(
                (dataset.mode_labels == mode_idx) & (dataset.turb_labels == turb_cat)
            )[0]
            samples = indices[:n_per_cell]

            for k, idx in enumerate(samples):
                img, _, _ = dataset[idx]
                img_in = img.unsqueeze(0).to(device)
                recon, _, _ = vae(img_in)

                orig_np = ((img.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)
                recon_np = ((recon.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)

                base_col = col_t * n_per_cell * 2 + k * 2
                axes[row, base_col].imshow(orig_np, cmap="hot")
                axes[row, base_col].axis("off")
                axes[row, base_col + 1].imshow(recon_np, cmap="hot")
                axes[row, base_col + 1].axis("off")

            if col_t == 0:
                axes[row, 0].set_ylabel(dataset.mode_display_name(mode_idx), fontsize=8)

        # Column headers (mode 0, first pair of each turbulence level)
        if row == 0:
            for col_t, turb_cat in enumerate(turb_cats):
                base_col = col_t * n_per_cell * 2
                axes[0, base_col].set_title(f"Turb {turb_cat}\nOrig", fontsize=7)
                axes[0, base_col + 1].set_title(f"Turb {turb_cat}\nRecon", fontsize=7)

    plt.suptitle("OAM VAE Reconstructions (Mode × Turbulence)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "reconstruction_grid.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─── Main entry point ────────────────────────────────────────────────────────

def visualize_oam(
    vae_checkpoint,
    mat_path,
    output_dir="vis_oam",
    device="cuda",
    tsne=True,
    pca_scatter=True,
    interpolation=True,
    traversal=True,
    reconstruction=True,
    max_samples=5000,
):
    """Run all latent space visualizations.

    Args:
        vae_checkpoint: Path to trained VAE checkpoint.
        mat_path: Path to OAM .mat file.
        output_dir: Directory to save all figures.
        tsne: If True, produce t-SNE scatter (slow for large datasets).
        pca_scatter: If True, produce PCA scatter (fast).
        interpolation: If True, produce interpolation plots for all mode pairs.
        traversal: If True, produce PCA traversal plot.
        reconstruction: If True, produce reconstruction grid.
        max_samples: Max samples for t-SNE (subsampled if larger).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    vae = _load_vae(vae_checkpoint, device)
    dataset = OAMDataset(mat_path)

    # Encode full dataset once
    print("\nEncoding dataset...")
    mus, mode_labels, turb_labels = encode_dataset(vae, dataset, device)
    print(f"Latent matrix shape: {mus.shape}")

    # 1. Scatter plots
    if pca_scatter:
        print("\n--- PCA scatter ---")
        plot_latent_scatter(mus, mode_labels, turb_labels, dataset, output_dir,
                            method="pca", max_samples=max_samples)
    if tsne:
        print("\n--- t-SNE scatter ---")
        plot_latent_scatter(mus, mode_labels, turb_labels, dataset, output_dir,
                            method="tsne", max_samples=max_samples)

    # 2. Interpolation — all adjacent mode pairs
    if interpolation:
        print("\n--- Interpolations ---")
        n_modes = len(dataset.modes)
        for i in range(n_modes):
            for j in range(i + 1, min(i + 3, n_modes)):  # nearest neighbours only
                plot_interpolation(vae, dataset, device, output_dir,
                                   n_steps=10, mode_a=i, mode_b=j)

    # 3. PCA traversal
    if traversal:
        print("\n--- PCA traversal ---")
        plot_pca_traversal(vae, mus, dataset, device, output_dir,
                           n_components=6, n_steps=7)

    # 4. Reconstruction grid
    if reconstruction:
        print("\n--- Reconstruction grid ---")
        plot_reconstruction_grid(vae, dataset, device, output_dir)

    print(f"\nAll visualizations saved to {output_dir}/")
