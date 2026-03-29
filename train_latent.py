"""
Training loop for DDPM in latent space.

Identical to the pixel-space training (train.py) but operates on
precomputed 32×32×4 latent tensors instead of 32×32×3 images.
Uses the same UNet architecture with in_channels=4.
"""

import copy
import os
import torch

try:
    import wandb
except ImportError:
    wandb = None

from diffusion import GaussianDiffusion
from dataset_hires import get_latent_dataloader
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


def train_latent(
    latent_path="data/celeba_latents.pt",
    batch_size=128,
    lr=2e-4,
    total_steps=500_000,
    save_dir="checkpoints_latent",
    save_every=50_000,
    log_every=1000,
    resume=None,
    device="cuda",
    num_workers=4,
    stable_diffusion_vae=False,
    prediction_target="epsilon",
    objective_type="l_simple",
    variance_mode="fixed",
    use_wandb=False,
    wandb_project="ddpm_mlmi4",
    wandb_run_name=None,
):
    """Train DDPM on precomputed latents.

    Args:
        latent_path: Path to precomputed latent tensors.
        batch_size: Batch size.
        lr: Learning rate.
        total_steps: Total training steps.
        save_dir: Checkpoint directory.
        save_every: Save checkpoint every N steps.
        log_every: Log loss every N steps.
        resume: Path to checkpoint to resume from.
        device: Device string.
        num_workers: DataLoader workers.
        stable_diffusion_vae: If True, use Stable Diffusion's VAE instead of the custom one.
        prediction_target: What the model predicts ("epsilon" or "mu").
        objective_type: Objective to optimize ("l_simple", "mse", or "l_vlb").
        variance_mode: Reverse variance mode ("fixed" or "learned").
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (auto-generated if None).
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it with `pip install wandb` or disable --wandb.")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=dict(
                latent_path=latent_path,
                batch_size=batch_size,
                lr=lr,
                total_steps=total_steps,
                save_every=save_every,
                log_every=log_every,
                stable_diffusion_vae=stable_diffusion_vae,
                prediction_target=prediction_target,
                objective_type=objective_type,
                variance_mode=variance_mode,
                device=str(device),
            ),
            tags=["train", "latent-space"],
        )

    # Model — same UNet but with 4 input channels for latents
    model = UNet(
        in_channels=4,
        out_channels=8 if variance_mode == "learned" else 4,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
        image_size=32,  # latent spatial size
    ).to(device)

    # Diffusion
    diffusion = GaussianDiffusion(
        T=1000,
        device=device,
        prediction_target=prediction_target,
        objective_type=objective_type,
        variance_mode=variance_mode,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # EMA
    ema = EMA(model, decay=0.9999)

    # Resume
    start_step = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        ckpt_prediction_target = checkpoint.get("prediction_target")
        ckpt_objective_type = checkpoint.get("objective_type")
        ckpt_variance_mode = checkpoint.get("variance_mode")

        if ckpt_prediction_target is not None and ckpt_prediction_target != prediction_target:
            raise ValueError(
                f"Resume checkpoint prediction_target={ckpt_prediction_target} does not match "
                f"requested prediction_target={prediction_target}."
            )
        if ckpt_objective_type is not None and ckpt_objective_type != objective_type:
            raise ValueError(
                f"Resume checkpoint objective_type={ckpt_objective_type} does not match "
                f"requested objective_type={objective_type}."
            )
        if ckpt_variance_mode is not None and ckpt_variance_mode != variance_mode:
            raise ValueError(
                f"Resume checkpoint variance_mode={ckpt_variance_mode} does not match "
                f"requested variance_mode={variance_mode}."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        start_step = checkpoint["step"]
        print(f"Resumed from step {start_step}")

    # Data
    dataloader = get_latent_dataloader(
        latent_path=latent_path, batch_size=batch_size, num_workers=num_workers
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Latent dataset: {len(dataloader.dataset)} samples")
    print(f"Training for {total_steps - start_step} steps on {device}")
    print(
        f"Ablation config: prediction_target={prediction_target}, "
        f"objective_type={objective_type}, variance_mode={variance_mode}"
    )
    if use_wandb:
        wandb.summary["param_count"] = param_count
        wandb.summary["dataset_size"] = len(dataloader.dataset)
        wandb.summary["prediction_target"] = prediction_target
        wandb.summary["objective_type"] = objective_type
        wandb.summary["variance_mode"] = variance_mode

    data_iter = iter(dataloader)
    model.train()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        # Get batch (cycle through dataset)
        try:
            z0 = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            z0 = next(data_iter)

        z0 = z0.to(device)

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (z0.shape[0],), device=device)

        # Compute loss
        loss = diffusion.p_losses(model, z0, t)

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
            ckpt_path = os.path.join(save_dir, f"latent_ckpt_{step + 1}.pt")
            torch.save(
                {
                    "step": step + 1,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "prediction_target": prediction_target,
                    "objective_type": objective_type,
                    "variance_mode": variance_mode,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            if use_wandb:
                wandb.summary["last_checkpoint"] = ckpt_path

    print("Latent diffusion training complete.")
    if use_wandb:
        wandb.finish()
