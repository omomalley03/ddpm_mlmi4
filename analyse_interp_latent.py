"""
Latent-space slerp interpolation between OAM modes.

Two interpolation modes:

1. Direct latent slerp (no DDPM):
   - Encode real images x_a, x_b → latent means μ_a, μ_b
   - Slerp: z_α = slerp(μ_a, μ_b, α)  for α ∈ [0..1]
   - Decode each z_α with the VAE decoder
   - Output: one row per pair

2. DDPM-style latent slerp (Ho et al. Section 4.3, in latent space):
   - Requires a trained latent DDPM (--ldm_checkpoint)
   - Noise μ_a and μ_b to t* using the same ε (DDPM forward process)
   - Slerp the noisy latents, then reverse-denoise from t* to 0
   - One row per t* value, columns = α steps
   - Directly comparable to pixel-space slerp from analyse_interpolation.py

Usage:
    # Direct latent slerp only:
    python analyse_interp_latent.py \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --mat_path /path/to/data.mat \\
        --output_dir analysis_interp_latent

    # With DDPM-style slerp:
    python analyse_interp_latent.py \\
        --vae_checkpoint checkpoints_vae_128/vae_oam_epoch100.pt \\
        --ldm_checkpoint checkpoints_ldm/ldm_ckpt_200000.pt \\
        --mat_path /path/to/data.mat \\
        --output_dir analysis_interp_latent
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae import VAE
from diffusion import GaussianDiffusion
from model import UNet
from dataset_oam import OAMDataset

# VAE config — must match retrained 128px VAE
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM = 4
VAE_BASE_CHANNELS = 64
VAE_IMAGE_SIZE = 128
LATENT_SIZE = VAE_IMAGE_SIZE // (2 ** len(VAE_CHANNEL_MULTS))  # 8

# Latent DDPM UNet config
LDM_CHANNEL_MULTS = (1, 2)
LDM_BASE_CHANNELS = 64
LDM_ATN_RES = (LATENT_SIZE // 2,)  # 4


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


def load_ldm(checkpoint_path, device):
    model = UNet(
        in_channels=VAE_LATENT_DIM,
        base_channels=LDM_BASE_CHANNELS,
        channel_mults=LDM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=LDM_ATN_RES,
        dropout=0.0,
        image_size=LATENT_SIZE,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded latent DDPM from step {ckpt['step']}")
    return model


def get_mode_sample(dataset, mode_name, turb_level=None, seed=None):
    """Return one image and its VAE-encode-ready tensor for a given mode."""
    modes = dataset.modes
    mode_idx = modes.index(mode_name)
    mask = np.array(dataset.mode_labels) == mode_idx
    if turb_level is not None:
        mask &= np.array(dataset.turb_labels) == turb_level
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"No samples for mode={mode_name}, turb_level={turb_level}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(indices)
    img, _, _ = dataset[idx]
    return img  # (1, H, W) in [-1, 1]


@torch.no_grad()
def direct_slerp_row(vae, diffusion, mu_a, mu_b, n_steps):
    """Slerp in latent space and decode. Returns list of decoded images."""
    images = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        z_alpha = diffusion.slerp(mu_a, mu_b, alpha)
        img = vae.decode(z_alpha)
        images.append(img.squeeze(0).cpu())
    return images


@torch.no_grad()
def ddpm_slerp_row(vae, ldm_model, diffusion, mu_a, mu_b, t_star, n_steps):
    """Ho et al. slerp in latent space at depth t_star. Returns decoded images."""
    noise = torch.randn_like(mu_a)
    t_tensor = torch.tensor([t_star], device=mu_a.device)
    z_a_noisy = diffusion.q_sample(mu_a, t_tensor, noise=noise)
    z_b_noisy = diffusion.q_sample(mu_b, t_tensor, noise=noise)

    images = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1)
        z_alpha = diffusion.slerp(z_a_noisy, z_b_noisy, alpha)
        z0 = diffusion.p_sample_loop_from_t(ldm_model, z_alpha, t_star)
        img = vae.decode(z0)
        images.append(img.squeeze(0).cpu())
    return images


def save_grid(rows, row_labels, col_labels, out_path, title=""):
    """Save an image grid where each entry is a (1, H, W) tensor in [-1, 1]."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.8))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for r, (row_imgs, row_lbl) in enumerate(zip(rows, row_labels)):
        for c, img in enumerate(row_imgs):
            vis = ((img.squeeze().numpy() + 1) / 2).clip(0, 1)
            axes[r, c].imshow(vis, cmap="hot", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, 0].set_ylabel(row_lbl, fontsize=8)
        if r == 0:
            for c, lbl in enumerate(col_labels):
                axes[0, c].set_title(lbl, fontsize=7)

    if title:
        plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(
    vae_checkpoint,
    mat_path,
    output_dir="analysis_interp_latent",
    ldm_checkpoint=None,
    mode_a="gauss",
    mode_b="p4",
    turb_level=None,
    n_steps=9,
    n_pairs=3,
    t_stars="250 500 750",
    device="cuda",
    modes=None,       # filter dataset to specific modes (must include mode_a and mode_b)
    turb_levels=None, # filter dataset to specific turb levels
):
    """
    modes / turb_levels control which images are loaded from the .mat file.
    They must be consistent with how the VAE and latent DDPM were trained.

    Model A: modes=["gauss"],                    turb_levels=[1,2,3]
    Model B: modes=["gauss","p1","p2","p3","p4"], turb_levels=[3]
    Original VAE: modes=["gauss","p4"],           turb_levels=None (all)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    t_star_list = [int(t) for t in str(t_stars).split()]

    vae = load_vae(vae_checkpoint, device)
    diffusion = GaussianDiffusion(T=1000, device=device)
    ldm_model = load_ldm(ldm_checkpoint, device) if ldm_checkpoint else None

    dataset = OAMDataset(mat_path, image_size=VAE_IMAGE_SIZE,
                         modes=modes, turb_levels=turb_levels)

    col_labels = [f"α={i/(n_steps-1):.1f}" for i in range(n_steps)]

    for pair_idx in range(n_pairs):
        seed_a = pair_idx * 2
        seed_b = pair_idx * 2 + 1
        img_a = get_mode_sample(dataset, mode_a, turb_level, seed=seed_a)
        img_b = get_mode_sample(dataset, mode_b, turb_level, seed=seed_b)

        img_a_t = img_a.unsqueeze(0).to(device)
        img_b_t = img_b.unsqueeze(0).to(device)

        with torch.no_grad():
            _, mu_a, _ = vae.encode(img_a_t)
            _, mu_b, _ = vae.encode(img_b_t)

        rows = []
        row_labels = []

        # Row 0: source images spaced as endpoints
        src_row = (
            [img_a.cpu()] +
            [torch.zeros_like(img_a)] * (n_steps - 2) +
            [img_b.cpu()]
        )
        rows.append(src_row)
        row_labels.append("source")

        # Row 1: direct latent slerp
        direct_row = direct_slerp_row(vae, diffusion, mu_a, mu_b, n_steps)
        rows.append(direct_row)
        row_labels.append("latent\nslerp")

        # Rows 2+: DDPM-style latent slerp at each t*
        if ldm_model is not None:
            for t_star in t_star_list:
                ddpm_row = ddpm_slerp_row(
                    vae, ldm_model, diffusion, mu_a, mu_b, t_star, n_steps
                )
                rows.append(ddpm_row)
                row_labels.append(f"t*={t_star}")

        turb_tag = f"_turb{turb_level}" if turb_level is not None else ""
        out_path = os.path.join(
            output_dir,
            f"latent_interp_{mode_a}_to_{mode_b}{turb_tag}_pair{pair_idx}.png",
        )
        save_grid(
            rows, row_labels, col_labels, out_path,
            title=f"Latent slerp: {mode_a} → {mode_b}{turb_tag}  (pair {pair_idx})",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--mat_path", required=True)
    parser.add_argument("--output_dir", default="analysis_interp_latent")
    parser.add_argument("--ldm_checkpoint", default=None,
                        help="Optional latent DDPM checkpoint for DDPM-style slerp rows")
    parser.add_argument("--mode_a", default="gauss")
    parser.add_argument("--mode_b", default="p4")
    parser.add_argument("--turb_level", type=int, default=None,
                        help="Fix turbulence level for both endpoints")
    parser.add_argument("--n_steps", type=int, default=9,
                        help="Number of interpolation steps (columns)")
    parser.add_argument("--n_pairs", type=int, default=3,
                        help="Number of image pairs to interpolate")
    parser.add_argument("--t_stars", type=str, default="250 500 750",
                        help="Space-separated t* values for DDPM-style slerp rows")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="Modes to load from dataset, must include mode_a and mode_b. "
                             "None = all modes. Model A: gauss. Model B: gauss p1 p2 p3 p4.")
    parser.add_argument("--turb_levels", type=int, nargs="+", default=None,
                        help="Turbulence levels to load, e.g. --turb_levels 1 2 3. "
                             "None = all levels. Model A: 1 2 3. Model B: 3.")
    args = parser.parse_args()
    main(**vars(args))
