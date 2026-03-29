"""
Training loop for DDPM with EMA (Ho et al. 2020).

- Adam optimizer (lr=2e-4)
- EMA with decay=0.9999
- Step-based training (not epoch-based)
- Periodic checkpointing
"""

import co
import os
import torch

from diffusion import GaussianDiffusion
from model import UNet

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt

from dataset_oam import get_oam_dataloader

OAM_CHANNEL_MULTS = (1, 2, 4, 4, 4)
OAM_LATENT_DIM = 4
OAM_BASE_CHANNELS = 64


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def train(
    mat_path,
    batch_size=128,
    lr=2e-4,
    total_steps=300_000,
    save_dir="checkpoints",
    save_every=50_000,
    log_every=1000,
    resume=None,
    device="cuda",
    image_size=128,
    num_workers=4,
    subset_size=None,
    turb_levels=None,
    modes=None,
):
    """Main training function.

    Args:
        dataset: Dataset name.
        batch_size: Batch size.
        lr: Learning rate.
        total_steps: Total training steps.
        save_dir: Checkpoint directory.
        save_every: Save checkpoint every N steps.
        log_every: Log loss every N steps.
        resume: Path to checkpoint to resume from.
        device: Device string.
        image_size: Image spatial size.
        num_workers: DataLoader workers.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet(
        in_channels=1,
        base_channels=OAM_BASE_CHANNELS,
        channel_mults=OAM_CHANNEL_MULTS,
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
        image_size=image_size,
    ).to(device)

    # Diffusion
    diffusion = GaussianDiffusion(T=1000, device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # EMA
    ema = EMA(model, decay=0.9999)

    # Resume
    start_step = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # Data
    dataloader, dataset = get_oam_dataloader(
        mat_path, batch_size=batch_size, num_workers=num_workers, image_size=image_size,
        turb_levels=turb_levels, modes=modes,
    )

    print(f"Dataset: {len(dataset)} images | Training on {device}")
    data_iter = iter(dataloader)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Training for {total_steps - start_step} steps on {device}")

    model.train()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        # Get batch (cycle through dataset)
        try:
            x0, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x0, _, _ = next(data_iter)

        x0 = x0.to(device)

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)

        # Compute loss (Algorithm 1)
        loss = diffusion.p_losses(model, x0, t)

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        ema.update(model)

        running_loss += loss.item()

        # Logging
        if (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            print(f"Step {step + 1}/{total_steps} | Loss: {avg_loss:.4f}")
            running_loss = 0.0

        # Checkpointing
        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            ckpt_path = os.path.join(save_dir, f"ckpt_{step + 1}.pt")
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")
