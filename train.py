"""
Training loop for DDPM with EMA (Ho et al. 2020).

- Adam optimizer (lr=2e-4)
- EMA with decay=0.9999
- Step-based training (not epoch-based)
- Periodic checkpointing
"""

import copy
import os
import torch

try:
    import wandb
except ImportError:
    wandb = None

from diffusion import GaussianDiffusion
from dataset import get_dataloader
from dataset_hires import get_hires_dataloader
from model import UNet


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


def _extract_images(batch):
    """Extract image tensor from a DataLoader batch.

    Supports datasets returning:
    - tensor images directly
    - tuples/lists like (images, labels)
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def train(
    dataset="cifar10",
    batch_size=128,
    lr=2e-4,
    total_steps=1_300_000,
    save_dir="checkpoints",
    save_every=50_000,
    log_every=1000,
    resume=None,
    device="cuda",
    image_size=32,
    num_workers=4,
    subset_size=None,
    use_wandb=False,
    wandb_project="ddpm_mlmi4",
    wandb_run_name=None,
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
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (auto-generated if None).
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with `pip install wandb` or disable --wandb.")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=dict(
                dataset=dataset,
                batch_size=batch_size,
                lr=lr,
                total_steps=total_steps,
                save_every=save_every,
                log_every=log_every,
                image_size=image_size,
                subset_size=subset_size,
                device=str(device),
            ),
            tags=["train", "pixel-space"],
        )

    # Model
    model = UNet(
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
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
    if dataset == "celeba_hq":
        dataloader = get_hires_dataloader(
            dataset="celeba_hq", image_size=image_size, batch_size=batch_size,
            data_dir="./data", num_workers=num_workers,
        )
    else:
        dataloader = get_dataloader(
            dataset=dataset, batch_size=batch_size, num_workers=num_workers,
            subset_size=subset_size,
        )
    data_iter = iter(dataloader)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Training for {total_steps - start_step} steps on {device}")
    if use_wandb:
        wandb.summary["param_count"] = param_count
        wandb.summary["dataset_size"] = len(dataloader.dataset)

    model.train()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x0 = _extract_images(batch)

        x0 = x0.to(device)

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)

        # Compute loss (Algorithm 1)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss = diffusion.p_losses(model, x0, t)

        # Gradient step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema.update(model)

        running_loss += loss.item()

        # Logging
        if (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            print(f"Step {step + 1}/{total_steps} | Loss: {avg_loss:.4f}")
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "progress/step": step + 1,
                        "progress/pct_complete": 100.0 * (step + 1) / total_steps,
                    },
                    step=step + 1,
                )
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
            if use_wandb:
                wandb.summary["last_checkpoint"] = ckpt_path

    print("Training complete.")
    if use_wandb:
        wandb.finish()
