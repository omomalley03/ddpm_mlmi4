"""
Time-bounded VAE training on OAM beam data (~1 hour on GPU).

Architecture: 320x320x1 → 5 downsamples → 10x10x4 latent
Loss: MSE reconstruction + KL divergence

Usage:
/home/omo26/rds/hpc-work/OAM/classify_environment/pupil_only/DATA_classify_environment_2_2_all_beams.mat
    python trialrun_train_vae_oam.py --mat_path /home/omo26/rds/hpc-work/OAM/classify_environment/pupil_only/DATA_classify_environment_2_2_all_beams.mat
    python trialrun_train_vae_oam.py --mat_path /home/omo26/rds/hpc-work/OAM/classify_environment/pupil_only/DATA_classify_environment_2_2_all_beams.mat --hours 1.0 --batch_size 16
"""

import argparse
import os
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vae import VAE
from dataset_oam import get_oam_dataloader

# 320→160→80→40→20→10 (5 stages), latent = 10×10×4
CHANNEL_MULTS = (1, 2, 4, 4, 4)
LATENT_DIM    = 4
BASE_CHANNELS = 64


def save_recon_grid(vae, fixed_batch, device, path, step):
    vae.eval()
    with torch.no_grad():
        imgs = fixed_batch[:8].to(device)
        recon, _, _ = vae(imgs)
    imgs  = ((imgs.cpu().squeeze(1).numpy()  + 1) / 2).clip(0, 1)
    recon = ((recon.cpu().squeeze(1).numpy() + 1) / 2).clip(0, 1)
    n = len(imgs)
    _, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(imgs[i],  cmap="hot", vmin=0, vmax=1)
        axes[1, i].imshow(recon[i], cmap="hot", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("real",  fontsize=8)
    axes[1, 0].set_ylabel("recon", fontsize=8)
    plt.suptitle(f"step {step}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    vae.train()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, dataset = get_oam_dataloader(
        args.mat_path,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} steps/epoch")

    fixed_batch, _, _ = next(iter(loader))

    vae = VAE(
        in_channels=1,
        base_channels=BASE_CHANNELS,
        channel_mults=CHANNEL_MULTS,
        latent_dim=LATENT_DIM,
    ).to(device)
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    t_start  = time.time()
    deadline = t_start + args.hours * 3600
    step = epoch = 0

    print(f"Training for {args.hours}h → {args.out_dir}")

    while time.time() < deadline:
        epoch += 1
        for imgs, _, _ in loader:
            if time.time() >= deadline:
                break

            imgs = imgs.to(device)
            recon, mu, logvar = vae(imgs)
            loss = VAE.recon_loss(recon, imgs) + args.kl_weight * VAE.kl_loss(mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 50 == 0:
                elapsed = (time.time() - t_start) / 60
                print(f"[{elapsed:5.1f}min | ep{epoch} step{step}] loss={loss.item():.4f}")

        snap = os.path.join(args.out_dir, f"recon_ep{epoch:03d}.png")
        save_recon_grid(vae, fixed_batch, device, snap, step)

    torch.save({
        "epoch": epoch, "step": step,
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(args.out_dir, "vae_final.pt"))
    print(f"Done — {epoch} epochs, {step} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path",   required=True)
    parser.add_argument("--out_dir",    default="vae_oam_run")
    parser.add_argument("--hours",      type=float, default=1.0)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--kl_weight",  type=float, default=1e-4)
    main(parser.parse_args())
