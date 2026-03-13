"""
VAE training on OAM laser beam images.

Architecture: 320×320×1 → (4 downsampling stages) → 20×20×8 latent
Loss: MSE reconstruction + KL divergence (kl_weight=1e-4)

Usage:
    python run.py --mode train_vae_oam --mat_path /path/to/data.mat
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt

from vae import VAE
from dataset_oam import get_oam_dataloader

# OAM VAE config: 320→160→80→40→20, latent = 20×20×8
OAM_CHANNEL_MULTS = (1, 2, 4, 4)
OAM_LATENT_DIM = 8
OAM_BASE_CHANNELS = 64


def train_vae_oam(
    mat_path,
    batch_size=32,
    lr=1e-4,
    total_epochs=100,
    kl_weight=1e-4,
    save_dir="checkpoints_vae_oam",
    save_every=10,
    log_every=50,
    resume=None,
    device="cuda",
    num_workers=4,
    image_size=None,  # None = keep original 320×320
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Model: grayscale (1 channel), 4 downsampling stages
    vae = VAE(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        latent_dim=OAM_LATENT_DIM,
    ).to(device)

    param_count = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {param_count:,}")
    print(f"Latent shape per image: ({OAM_LATENT_DIM}, 20, 20) = {OAM_LATENT_DIM*20*20} dims")

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    start_epoch = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["vae"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    dataloader, dataset = get_oam_dataloader(
        mat_path, batch_size=batch_size, num_workers=num_workers, image_size=image_size,
    )
    print(f"Dataset: {len(dataset)} images | Training on {device}")

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, total_epochs):
        vae.train()
        epoch_recon = epoch_kl = 0.0
        n_steps = 0

        for imgs, mode_labels, turb_labels in dataloader:
            imgs = imgs.to(device)

            x_recon, mu, logvar = vae(imgs)
            recon = VAE.recon_loss(x_recon, imgs)
            kl = VAE.kl_loss(mu, logvar)
            loss = recon + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                print(f"Epoch {epoch+1}/{total_epochs} | Step {global_step} | "
                      f"Recon: {recon.item():.4f} | KL: {kl.item():.2f}")

        avg_recon = epoch_recon / n_steps
        avg_kl = epoch_kl / n_steps
        print(f"Epoch {epoch+1}/{total_epochs} done | Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.2f}")

        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            # Checkpoint
            ckpt_path = os.path.join(save_dir, f"vae_oam_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "vae": vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "kl_weight": kl_weight,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

            # Reconstruction comparison grid per mode
            _save_recon_grid(vae, dataset, device, save_dir, epoch + 1)


def _save_recon_grid(vae, dataset, device, save_dir, epoch):
    """Save a grid of original vs reconstructed images, one row per OAM mode."""
    vae.eval()
    n_modes = len(dataset.modes)
    n_cols = 6  # pairs: 3 originals + 3 reconstructions side by side

    fig, axes = plt.subplots(n_modes, n_cols * 2, figsize=(n_cols * 4, n_modes * 2.5))
    if n_modes == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row, mode_idx in enumerate(range(n_modes)):
            # Find n_cols samples from this mode
            mode_indices = (dataset.mode_labels == mode_idx).nonzero()[0]
            sample_indices = mode_indices[: n_cols]

            for col, idx in enumerate(sample_indices):
                img, _, _ = dataset[idx]
                img_in = img.unsqueeze(0).to(device)
                recon, _, _ = vae(img_in)

                # Convert to numpy for display: [-1,1] → [0,1]
                orig_np = ((img.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)
                recon_np = ((recon.squeeze().cpu().numpy() + 1) / 2).clip(0, 1)

                axes[row, col * 2].imshow(orig_np, cmap="hot")
                axes[row, col * 2].axis("off")
                axes[row, col * 2 + 1].imshow(recon_np, cmap="hot")
                axes[row, col * 2 + 1].axis("off")

                if col == 0:
                    axes[row, 0].set_ylabel(dataset.mode_display_name(mode_idx), fontsize=9)

            # Column headers on first row only
            if row == 0:
                for col in range(n_cols):
                    axes[0, col * 2].set_title("Original", fontsize=7)
                    axes[0, col * 2 + 1].set_title("Recon", fontsize=7)

    plt.suptitle(f"OAM VAE Reconstructions — Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"recon_epoch{epoch}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction grid: {out_path}")
