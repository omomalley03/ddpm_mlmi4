"""
Compare interpolation behavior between pixel-space DDPM and latent-space DDPM.

For each pair:
1) Sample two endpoint noises in each model's native space
2) Linearly interpolate between endpoints across n_interp points
3) Run reverse diffusion from each interpolated noise
4) Save three figures:
   - pixel interpolation row
   - latent interpolation row (decoded to RGB)
   - 2-row comparison (top=pixel, bottom=latent)

Example:
    python compare_interpolation.py \
        --pixel_checkpoint checkpoints_celebahq/ckpt_500000.pt \
        --latent_checkpoint checkpoints_latent/latent_ckpt_200000.pt \
        --vae_checkpoint checkpoints_vae/vae_epoch50.pt \
        --output_dir interpolation_compare \
        --n_pairs 4 --n_interp 9
"""

import argparse
import os

import torch
from torchvision.utils import make_grid, save_image

from diffusers import AutoencoderKL

from diffusion import GaussianDiffusion
from model import UNet
from vae import VAE


SD_VAE_REPO = "stabilityai/sd-vae-ft-ema"


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


@torch.no_grad()
def p_sample_from_noise(diffusion, model, x_t):
    x = x_t
    for t in reversed(range(diffusion.T)):
        x = diffusion.p_sample(model, x, t)
    return x


@torch.no_grad()
def p_sample_to_t(diffusion, model, x_start, target_t):
    # x_start is assumed to be at t = diffusion.T - 1 (pure noise start)
    x = x_start
    start_t = diffusion.T - 1
    for t in range(start_t, target_t, -1):
        x = diffusion.p_sample(model, x, t)
    return x  # x at target_t


@torch.no_grad()
def p_sample_from_t(diffusion, model, x_t, t_start):
    # Denoise from x_t down to x_0
    x = x_t
    for t in range(t_start, -1, -1):
        x = diffusion.p_sample(model, x, t)
    return x


def denorm_to_01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def load_pixel_model(checkpoint_path, image_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=image_size,
    ).to(device)

    model.load_state_dict(checkpoint["ema"])
    model.eval()

    diffusion = GaussianDiffusion(
        T=1000,
        device=device,
        prediction_target=checkpoint.get("prediction_target", "epsilon"),
        objective_type=checkpoint.get("objective_type", "l_simple"),
        variance_mode=checkpoint.get("variance_mode", "fixed"),
    )

    return model, diffusion, checkpoint.get("step", "unknown")


def load_latent_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    prediction_target = checkpoint.get("prediction_target", "epsilon")
    objective_type = checkpoint.get("objective_type", "l_simple")
    variance_mode = checkpoint.get("variance_mode", "fixed")

    model = UNet(
        in_channels=4,
        out_channels=8 if variance_mode == "learned" else 4,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        image_size=32,
    ).to(device)

    model.load_state_dict(checkpoint["ema"])
    model.eval()

    diffusion = GaussianDiffusion(
        T=1000,
        device=device,
        prediction_target=prediction_target,
        objective_type=objective_type,
        variance_mode=variance_mode,
    )

    return model, diffusion, checkpoint.get("step", "unknown")


def load_decoder(vae_checkpoint, use_stable_diffusion_vae, device):
    if use_stable_diffusion_vae:
        vae = AutoencoderKL.from_pretrained(SD_VAE_REPO).to(device)
        vae.eval()
        return vae

    vae = VAE(in_channels=3, base_channels=64, latent_dim=4).to(device)
    checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["vae"])
    vae.eval()
    return vae


@torch.no_grad()
def decode_latents(vae, latents, use_stable_diffusion_vae):
    if use_stable_diffusion_vae:
        return vae.decode(latents / vae.config.scaling_factor).sample
    return vae.decode(latents)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare interpolation: pixel DDPM vs latent DDPM")
    parser.add_argument("--pixel_checkpoint", type=str, required=True)
    parser.add_argument("--latent_checkpoint", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--use_stable_diffusion_vae", action="store_true", default=False)

    parser.add_argument("--pixel_image_size", type=int, default=256)
    parser.add_argument("--n_pairs", type=int, default=4)
    parser.add_argument("--n_interp", type=int, default=9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--interp_t_frac",
        type=float,
        default=0.125,
        help="Fraction of diffusion horizon to interpolate at (0..1), e.g. 0.125 = T/8",
    )

    parser.add_argument("--output_dir", type=str, default="interpolation_compare")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not args.use_stable_diffusion_vae and args.vae_checkpoint is None:
        parser.error("--vae_checkpoint is required unless --use_stable_diffusion_vae is set")
    if not (0.0 <= args.interp_t_frac <= 1.0):
        parser.error("--interp_t_frac must be in [0, 1]")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading pixel-space model...")
    pixel_model, pixel_diffusion, pixel_step = load_pixel_model(
        args.pixel_checkpoint,
        image_size=args.pixel_image_size,
        device=device,
    )

    print("Loading latent-space model...")
    latent_model, latent_diffusion, latent_step = load_latent_model(
        args.latent_checkpoint,
        device=device,
    )

    print("Loading VAE decoder...")
    vae = load_decoder(args.vae_checkpoint, args.use_stable_diffusion_vae, device)

    print(
        f"Creating interpolation figures: pairs={args.n_pairs}, points={args.n_interp}, "
        f"pixel_step={pixel_step}, latent_step={latent_step}"
    )

    ts = torch.linspace(0.0, 1.0, args.n_interp, device=device)

    if pixel_diffusion.T != latent_diffusion.T:
        raise ValueError(
            f"Pixel/latent diffusion T mismatch: {pixel_diffusion.T} vs {latent_diffusion.T}"
        )
    interp_t = int(round((pixel_diffusion.T - 1) * args.interp_t_frac))
    print(
        f"Interpolating at diffusion timestep t={interp_t} "
        f"(T={pixel_diffusion.T}, frac={args.interp_t_frac:.4f})"
    )

    for pair_idx in range(args.n_pairs):
        pair_seed = args.seed + pair_idx
        torch.manual_seed(pair_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pair_seed)

        # Endpoint noises in each model space
        pixel_noise_a = torch.randn(1, 3, args.pixel_image_size, args.pixel_image_size, device=device)
        pixel_noise_b = torch.randn(1, 3, args.pixel_image_size, args.pixel_image_size, device=device)

        latent_noise_a = torch.randn(1, 4, 32, 32, device=device)
        latent_noise_b = torch.randn(1, 4, 32, 32, device=device)

        # Move each endpoint from pure noise down to t = T//8
        pixel_xt_a = p_sample_to_t(pixel_diffusion, pixel_model, pixel_noise_a, interp_t)
        pixel_xt_b = p_sample_to_t(pixel_diffusion, pixel_model, pixel_noise_b, interp_t)

        latent_xt_a = p_sample_to_t(latent_diffusion, latent_model, latent_noise_a, interp_t)
        latent_xt_b = p_sample_to_t(latent_diffusion, latent_model, latent_noise_b, interp_t)

        pixel_images = []
        latent_images = []

        print(f"Pair {pair_idx + 1}/{args.n_pairs}: sampling interpolation trajectory...")

        for t in ts:
            # Pixel DDPM interpolation at t=T//8, then denoise to x0
            x_t_pixel = lerp(pixel_xt_a, pixel_xt_b, t)
            pixel_sample = p_sample_from_t(pixel_diffusion, pixel_model, x_t_pixel, interp_t)
            pixel_images.append(denorm_to_01(pixel_sample))

            # Latent DDPM interpolation at t=T//8, then denoise + decode
            x_t_latent = lerp(latent_xt_a, latent_xt_b, t)
            latent_sample = p_sample_from_t(latent_diffusion, latent_model, x_t_latent, interp_t)
            latent_rgb = decode_latents(vae, latent_sample, args.use_stable_diffusion_vae)
            latent_images.append(denorm_to_01(latent_rgb))

        pixel_row = torch.cat(pixel_images, dim=0)
        latent_row = torch.cat(latent_images, dim=0)

        pixel_grid = make_grid(pixel_row, nrow=args.n_interp, padding=2)
        latent_grid = make_grid(latent_row, nrow=args.n_interp, padding=2)
        comparison_grid = make_grid(torch.cat([pixel_row, latent_row], dim=0), nrow=args.n_interp, padding=2)

        pair_name = f"pair_{pair_idx + 1:02d}"
        pixel_path = os.path.join(args.output_dir, f"{pair_name}_pixel.png")
        latent_path = os.path.join(args.output_dir, f"{pair_name}_latent.png")
        comparison_path = os.path.join(args.output_dir, f"{pair_name}_comparison.png")

        save_image(pixel_grid, pixel_path)
        save_image(latent_grid, latent_path)
        save_image(comparison_grid, comparison_path)

        print(f"Saved: {pixel_path}")
        print(f"Saved: {latent_path}")
        print(f"Saved: {comparison_path}")

    print("Done.")


if __name__ == "__main__":
    main()
