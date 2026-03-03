"""
Training loop for the VAE on CelebA-HQ 256×256.

Trains encoder/decoder to compress images to 32×32×4 latents.
Loss: MSE reconstruction + KL divergence (weighted by kl_weight).
"""

import os
import torch
from torchvision.utils import save_image

from vae import VAE
from dataset_hires import get_hires_dataloader


def train_vae(
    dataset="celeba_hq",
    image_size=256,
    batch_size=16,
    lr=1e-4,
    total_epochs=50,
    kl_weight=1e-4,
    save_dir="checkpoints_vae",
    save_every=10,
    log_every=100,
    resume=None,
    device="cuda",
    data_dir="./data",
    num_workers=4,
):
    """Train the VAE.

    Args:
        dataset: Dataset name.
        image_size: Input image size (256).
        batch_size: Batch size (16 for 256×256 on A100).
        lr: Learning rate.
        total_epochs: Number of training epochs.
        kl_weight: Weight for KL divergence term.
        save_dir: Checkpoint directory.
        save_every: Save checkpoint every N epochs.
        log_every: Log loss every N steps.
        resume: Path to checkpoint to resume from.
        device: Device string.
        data_dir: Data directory.
        num_workers: DataLoader workers.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Model
    vae = VAE(in_channels=3, base_channels=64, latent_dim=4).to(device)

    param_count = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Resume
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["vae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    # Data
    dataloader = get_hires_dataloader(
        dataset=dataset, image_size=image_size, batch_size=batch_size,
        data_dir=data_dir, num_workers=num_workers,
    )
    print(f"Dataset: {len(dataloader.dataset)} images")
    print(f"Training for {total_epochs - start_epoch} epochs on {device}")

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, total_epochs):
        vae.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_steps = 0

        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            x_recon, mu, logvar = vae(x)
            recon = VAE.recon_loss(x_recon, x)
            kl = VAE.kl_loss(mu, logvar)
            loss = recon + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()
            epoch_kl += kl.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                print(f"Epoch {epoch+1}/{total_epochs} | Step {global_step} | "
                      f"Recon: {recon.item():.4f} | KL: {kl.item():.2f} | "
                      f"Loss: {loss.item():.4f}")

        # Epoch summary
        avg_recon = epoch_recon / epoch_steps
        avg_kl = epoch_kl / epoch_steps
        print(f"Epoch {epoch+1}/{total_epochs} complete | "
              f"Avg Recon: {avg_recon:.4f} | Avg KL: {avg_kl:.2f}")

        # Save reconstruction samples at end of each epoch
        if (epoch + 1) % save_every == 0 or (epoch + 1) == total_epochs:
            # Save checkpoint
            ckpt_path = os.path.join(save_dir, f"vae_epoch{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "vae": vae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

            # Save reconstruction comparison
            vae.eval()
            with torch.no_grad():
                x_sample = x[:8]
                x_recon_sample, _, _ = vae(x_sample)
                comparison = torch.cat([x_sample, x_recon_sample], dim=0)
                comparison = (comparison + 1.0) / 2.0  # [-1,1] → [0,1]
                comparison = comparison.clamp(0.0, 1.0)
                recon_path = os.path.join(save_dir, f"recon_epoch{epoch+1}.png")
                save_image(comparison, recon_path, nrow=8)
                print(f"Saved reconstruction samples: {recon_path}")

    print("VAE training complete.")
