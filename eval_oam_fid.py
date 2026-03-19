"""
FID evaluation for OAM image generation.

Compares generated samples from one or both pipelines against real OAM images:
  - Pixel DDPM: UNet trained directly on 128×128 OAM images
  - LDM:        Latent DDPM on VAE latents, decoded back to image space

Key differences from CIFAR eval.py:
  - OAM images are grayscale (1 channel) → repeated to 3ch for InceptionV3
  - Real images loaded from OAMDataset (.mat), not CIFAR datasets
  - IS is skipped (meaningless for physics images — not ImageNet classes)
  - modes / turb_levels filter real and generated sets to the training distribution

Usage:
    # Pixel DDPM only:
    python eval_oam_fid.py \\
        --pixel_checkpoint checkpoints_gauss_turb3/ckpt_300000.pt \\
        --mat_path croped_2_2_pupil_data.mat \\
        --modes gauss --turb_levels 1 2 3

    # LDM only:
    python eval_oam_fid.py \\
        --vae_checkpoint checkpoints_vae_128_modelA/vae_oam_epoch100.pt \\
        --ldm_checkpoint checkpoints_ldm_modelA/ldm_ckpt_200000.pt \\
        --mat_path croped_2_2_pupil_data.mat \\
        --modes gauss --turb_levels 1 2 3

    # Both pipelines in one run (produces side-by-side comparison):
    python eval_oam_fid.py \\
        --pixel_checkpoint checkpoints_gauss_turb3/ckpt_300000.pt \\
        --vae_checkpoint  checkpoints_vae_128_modelA/vae_oam_epoch100.pt \\
        --ldm_checkpoint  checkpoints_ldm_modelA/ldm_ckpt_200000.pt \\
        --mat_path croped_2_2_pupil_data.mat \\
        --modes gauss --turb_levels 1 2 3
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse Inception helpers and FID calculation directly from the CIFAR eval
from eval import load_inception, get_inception_outputs, compute_fid

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE
from dataset_oam import OAMDataset

# ── Architecture constants — must match training scripts ───────────────────────

# Pixel-space OAM DDPM (train_ddpm_oam.py / sample_ddpm_oam.py)
OAM_CHANNEL_MULTS = (1, 2, 4, 4, 4)
OAM_BASE_CHANNELS  = 64

# Latent DDPM (train_ddpm_latent.py)
LDM_CHANNEL_MULTS = (1, 2)
LDM_BASE_CHANNELS  = 64

# VAE (train_vae_oam.py at 128px)
VAE_CHANNEL_MULTS = (1, 2, 4, 4)
VAE_LATENT_DIM    = 4
VAE_IMAGE_SIZE    = 128
LATENT_SIZE       = VAE_IMAGE_SIZE // (2 ** len(VAE_CHANNEL_MULTS))  # 8


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_pixel_ddpm(checkpoint_path, image_size, device):
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
    print(f"Loaded pixel DDPM from step {ckpt['step']}")
    return model, ckpt["step"]


def load_vae(checkpoint_path, device):
    vae = VAE(
        in_channels=1,
        base_channels=VAE_BASE_CHANNELS if "VAE_BASE_CHANNELS" in dir() else 64,
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
        attn_resolutions=(LATENT_SIZE // 2,),
        dropout=0.0,
        image_size=LATENT_SIZE,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    print(f"Loaded latent DDPM from step {ckpt['step']}")
    return model, ckpt["step"]


# ── Image preparation ─────────────────────────────────────────────────────────

def to_3ch_01(images_1ch):
    """Convert grayscale (N,1,H,W) in [-1,1] → (N,3,H,W) in [0,1].

    InceptionV3 expects 3-channel input in [0,1]. Repeating the single
    grayscale channel is the standard approach for FID on grey images.
    """
    images = (images_1ch + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    return images.repeat(1, 3, 1, 1)


# ── Sample generation ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate_pixel_ddpm(model, diffusion, n_eval, image_size, batch_size, device):
    """Generate n_eval samples from pixel-space DDPM."""
    all_samples = []
    generated = 0
    while generated < n_eval:
        this_batch = min(batch_size, n_eval - generated)
        samples = diffusion.p_sample_loop(
            model, (this_batch, 1, image_size, image_size)
        )
        all_samples.append(to_3ch_01(samples.cpu()))
        generated += this_batch
        print(f"  Generated {generated}/{n_eval}")
    return torch.cat(all_samples, dim=0)  # (N, 3, H, W)


@torch.no_grad()
def generate_ldm(vae, ldm_model, diffusion, n_eval, batch_size, device):
    """Generate n_eval samples via latent DDPM → VAE decode."""
    all_samples = []
    generated = 0
    while generated < n_eval:
        this_batch = min(batch_size, n_eval - generated)
        latents = diffusion.p_sample_loop(
            ldm_model, (this_batch, VAE_LATENT_DIM, LATENT_SIZE, LATENT_SIZE)
        )
        imgs = vae.decode(latents)           # (B, 1, 128, 128) in [-1,1]
        all_samples.append(to_3ch_01(imgs.cpu()))
        generated += this_batch
        print(f"  Generated {generated}/{n_eval}")
    return torch.cat(all_samples, dim=0)  # (N, 3, H, W)


# ── Real image loading ────────────────────────────────────────────────────────

def load_real_images(mat_path, image_size, modes, turb_levels, batch_size):
    """Load all real OAM images matching the given mode/turb filter."""
    dataset = OAMDataset(mat_path, image_size=image_size,
                         modes=modes, turb_levels=turb_levels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    imgs_list = []
    for imgs, _, _ in loader:
        imgs_list.append(to_3ch_01(imgs))   # (B, 1, H, W) → (B, 3, H, W)
    real_3ch = torch.cat(imgs_list, dim=0)
    print(f"Loaded {len(real_3ch)} real images  "
          f"(modes={dataset.modes}, turb_categories={dataset.turb_categories})")
    return real_3ch, dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FID evaluation for OAM pixel DDPM and/or LDM pipeline"
    )
    parser.add_argument("--mat_path", required=True,
                        help="Path to OAM .mat data file")
    parser.add_argument("--pixel_checkpoint", default=None,
                        help="Pixel-space OAM DDPM checkpoint (EMA weights)")
    parser.add_argument("--vae_checkpoint", default=None,
                        help="VAE checkpoint (required when --ldm_checkpoint is set)")
    parser.add_argument("--ldm_checkpoint", default=None,
                        help="Latent DDPM checkpoint (EMA weights)")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="OAM modes to evaluate, e.g. --modes gauss. "
                             "None = all modes. Model A: gauss. Model B: gauss p1 p2 p3 p4.")
    parser.add_argument("--turb_levels", type=int, nargs="+", default=None,
                        help="Turbulence levels to evaluate, e.g. --turb_levels 1 2 3. "
                             "None = all levels. Model A: 1 2 3. Model B: 3.")
    parser.add_argument("--n_eval", type=int, default=None,
                        help="Number of samples to generate per pipeline. "
                             "Default: match number of real images.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--output_dir", default="eval_oam_fid")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.pixel_checkpoint is None and args.ldm_checkpoint is None:
        parser.error("Provide at least one of --pixel_checkpoint or --ldm_checkpoint.")
    if args.ldm_checkpoint is not None and args.vae_checkpoint is None:
        parser.error("--vae_checkpoint is required when --ldm_checkpoint is set.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Real images ───────────────────────────────────────────────────────────
    print("\nLoading real OAM images...")
    real_3ch, dataset = load_real_images(
        args.mat_path, args.image_size, args.modes, args.turb_levels, args.batch_size
    )
    n_real = len(real_3ch)
    n_eval = args.n_eval if args.n_eval is not None else n_real
    print(f"n_real={n_real} | n_eval per pipeline={n_eval}")

    diffusion = GaussianDiffusion(T=1000, device=device)

    # ── Inception ─────────────────────────────────────────────────────────────
    print("\nLoading InceptionV3...")
    inception, pool3_features = load_inception(device)

    print("\nExtracting real image features...")
    real_feats, _ = get_inception_outputs(
        real_3ch, inception, pool3_features,
        batch_size=args.batch_size, device=device
    )

    # ── Results accumulator ───────────────────────────────────────────────────
    results_lines = [
        f"Real images:  N={n_real}  "
        f"modes={args.modes or 'all'}  "
        f"turb_levels={args.turb_levels or 'all'}",
        "",
    ]

    # ── LDM (VAE + latent DDPM) — runs first ─────────────────────────────────
    if args.ldm_checkpoint is not None:
        print(f"\n{'='*55}")
        print("Pipeline: LDM (VAE + latent DDPM)")
        print(f"{'='*55}")
        vae = load_vae(args.vae_checkpoint, device)
        ldm_model, ldm_step = load_ldm(args.ldm_checkpoint, device)

        print(f"\nGenerating {n_eval} LDM samples...")
        gen_ldm = generate_ldm(
            vae, ldm_model, diffusion, n_eval, args.batch_size, device
        )
        print("\nExtracting LDM features...")
        gen_feats_ldm, _ = get_inception_outputs(
            gen_ldm, inception, pool3_features,
            batch_size=args.batch_size, device=device
        )
        fid_ldm = compute_fid(real_feats, gen_feats_ldm)
        print(f"\nLDM FID: {fid_ldm:.4f}")

        results_lines += [
            "--- LDM (VAE + latent DDPM) ---",
            f"VAE:         {args.vae_checkpoint}",
            f"LDM:         {args.ldm_checkpoint}  (step={ldm_step})",
            f"N_generated: {n_eval}",
            f"FID:         {fid_ldm:.4f}",
            "",
        ]
        del vae, ldm_model, gen_ldm   # free VRAM before next pipeline

    # ── Pixel DDPM ────────────────────────────────────────────────────────────
    if args.pixel_checkpoint is not None:
        print(f"\n{'='*55}")
        print("Pipeline: Pixel DDPM")
        print(f"{'='*55}")
        pixel_model, step = load_pixel_ddpm(
            args.pixel_checkpoint, args.image_size, device
        )
        print(f"\nGenerating {n_eval} pixel DDPM samples...")
        gen_pixel = generate_pixel_ddpm(
            pixel_model, diffusion, n_eval, args.image_size, args.batch_size, device
        )
        print("\nExtracting pixel DDPM features...")
        gen_feats_pixel, _ = get_inception_outputs(
            gen_pixel, inception, pool3_features,
            batch_size=args.batch_size, device=device
        )
        fid_pixel = compute_fid(real_feats, gen_feats_pixel)
        print(f"\nPixel DDPM FID: {fid_pixel:.4f}")

        results_lines += [
            "--- Pixel DDPM ---",
            f"Checkpoint:  {args.pixel_checkpoint}  (step={step})",
            f"N_generated: {n_eval}",
            f"FID:         {fid_pixel:.4f}",
            "",
        ]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    for line in results_lines:
        print(line)
    print('='*55)

    results_path = os.path.join(args.output_dir, "fid_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(results_lines) + "\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
