"""
FID evaluation of interpolated samples from pixel-space and latent-space DDPMs.

Generates N interpolated images from each model by sampling random endpoint pairs
and interpolating at a fixed diffusion timestep, then computes FID against real
CelebA-HQ images. Ablates over multiple interp_t_frac values.

Example:
    python fid_interpolation.py \
        --pixel_checkpoint checkpoints_celebahq/ckpt_400000.pt \
        --latent_checkpoint checkpoints_latent/latent_ckpt_200000.pt \
        --use_stable_diffusion_vae \
        --n_images 2048 \
        --interp_t_fracs 0.0625 0.125 0.25 0.5 0.75 \
        --data_dir ./data \
        --output_dir fid_interpolation_results
"""

import argparse
import math
import os

import torch

from compare_interpolation import (
    decode_latents,
    denorm_to_01,
    lerp,
    load_decoder,
    load_latent_model,
    load_pixel_model,
    p_sample_from_t,
    p_sample_to_t,
)
from dataset_hires import get_hires_dataloader
from eval import compute_fid, get_inception_outputs, load_inception


@torch.no_grad()
def generate_interpolated_images(
    model, diffusion, spatial_shape, n_images, n_interp, interp_t, seed, device
):
    """Generate n_images interpolated samples.

    Batches both endpoints through p_sample_to_t together and all n_interp
    lerp points through p_sample_from_t together, minimising sequential passes.

    Returns (N, C, H, W) in [-1, 1] on CPU.
    """
    C, H, W = spatial_shape
    n_pairs = math.ceil(n_images / n_interp)
    ts = torch.linspace(0.0, 1.0, n_interp, device=device)
    chunks = []

    for pair_idx in range(n_pairs):
        torch.manual_seed(seed + pair_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + pair_idx)

        # Both endpoints in one forward pass: (2, C, H, W)
        noise_ab = torch.randn(2, C, H, W, device=device)
        xt_ab = p_sample_to_t(diffusion, model, noise_ab, interp_t)

        # All n_interp lerps in one batch: (n_interp, C, H, W)
        x_ts = torch.stack([lerp(xt_ab[0:1], xt_ab[1:2], t) for t in ts]).squeeze(1)
        samples = p_sample_from_t(diffusion, model, x_ts, interp_t)
        chunks.append(samples.cpu())

        n_so_far = min((pair_idx + 1) * n_interp, n_images)
        print(f"  Pair {pair_idx + 1}/{n_pairs}: {n_so_far}/{n_images}")

    return torch.cat(chunks)[:n_images]  # (N, C, H, W)


def load_real_images(data_dir, n_images, batch_size, image_size, num_workers):
    """Load n_images real CelebA-HQ images, returned in [0, 1]."""
    loader = get_hires_dataloader(
        dataset="celeba_hq",
        image_size=image_size,
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=num_workers,
    )
    chunks = []
    for batch in loader:
        chunks.append(denorm_to_01(batch).cpu())
        if sum(x.shape[0] for x in chunks) >= n_images:
            break
    images = torch.cat(chunks)[:n_images]
    print(f"Loaded {len(images)} real CelebA-HQ images")
    return images


def decode_latent_batch(raw_latents, vae, use_stable_diffusion_vae, batch_size, device):
    """Decode (N, 4, h, w) latents in [-1,1] to (N, 3, H, W) in [0,1] on CPU."""
    out = []
    for i in range(0, len(raw_latents), batch_size):
        batch = raw_latents[i:i + batch_size].to(device)
        rgb = decode_latents(vae, batch, use_stable_diffusion_vae)
        out.append(denorm_to_01(rgb).cpu())
    return torch.cat(out)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="FID of interpolated DDPM samples vs real CelebA-HQ")
    parser.add_argument("--pixel_checkpoint", type=str, required=True)
    parser.add_argument("--latent_checkpoint", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--use_stable_diffusion_vae", action="store_true", default=False)

    parser.add_argument("--pixel_image_size", type=int, default=256)
    parser.add_argument("--n_images", type=int, default=2048,
                        help="Number of interpolated images to generate per model per t_frac")
    parser.add_argument("--n_interp", type=int, default=9,
                        help="Interpolation points per pair")
    parser.add_argument("--interp_t_fracs", type=float, nargs="+",
                        default=[0.0625, 0.125, 0.25, 0.5, 0.75],
                        help="Denoising step fractions to ablate over")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="fid_interpolation_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not args.use_stable_diffusion_vae and args.vae_checkpoint is None:
        parser.error("--vae_checkpoint is required unless --use_stable_diffusion_vae is set")
    for frac in args.interp_t_fracs:
        if not (0.0 <= frac <= 1.0):
            parser.error(f"--interp_t_fracs values must be in [0, 1], got {frac}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading pixel-space model...")
    pixel_model, pixel_diffusion, pixel_step = load_pixel_model(
        args.pixel_checkpoint, image_size=args.pixel_image_size, device=device
    )
    print("Loading latent-space model...")
    latent_model, latent_diffusion, latent_step = load_latent_model(
        args.latent_checkpoint, device=device
    )
    print("Loading VAE decoder...")
    vae = load_decoder(args.vae_checkpoint, args.use_stable_diffusion_vae, device)

    # Real images and their inception features are shared across all t_fracs
    print("\nLoading real CelebA-HQ images...")
    real_images = load_real_images(
        args.data_dir, args.n_images, args.batch_size, args.pixel_image_size, args.num_workers
    )
    print("\nLoading InceptionV3...")
    inception, pool3_features = load_inception(device)
    print("Extracting features for real images...")
    real_feats, _ = get_inception_outputs(
        real_images, inception, pool3_features, batch_size=args.batch_size, device=device
    )

    results = []

    for frac in sorted(args.interp_t_fracs):
        interp_t = int(round((pixel_diffusion.T - 1) * frac))
        print(f"\n--- interp_t_frac={frac:.4f}  (t={interp_t}, T={pixel_diffusion.T}) ---")

        print("Generating pixel-space interpolations...")
        pixel_raw = generate_interpolated_images(
            pixel_model, pixel_diffusion,
            spatial_shape=(3, args.pixel_image_size, args.pixel_image_size),
            n_images=args.n_images, n_interp=args.n_interp,
            interp_t=interp_t, seed=args.seed, device=device,
        )
        pixel_images = denorm_to_01(pixel_raw)

        print("Generating latent-space interpolations...")
        latent_raw = generate_interpolated_images(
            latent_model, latent_diffusion,
            spatial_shape=(4, 32, 32),
            n_images=args.n_images, n_interp=args.n_interp,
            interp_t=interp_t, seed=args.seed, device=device,
        )
        print("Decoding latents...")
        latent_images = decode_latent_batch(
            latent_raw, vae, args.use_stable_diffusion_vae, args.batch_size, device
        )

        print("Extracting inception features...")
        pixel_feats, _ = get_inception_outputs(
            pixel_images, inception, pool3_features, batch_size=args.batch_size, device=device
        )
        latent_feats, _ = get_inception_outputs(
            latent_images, inception, pool3_features, batch_size=args.batch_size, device=device
        )

        pixel_fid = compute_fid(real_feats, pixel_feats)
        latent_fid = compute_fid(real_feats, latent_feats)
        print(f"  Pixel FID:  {pixel_fid:.2f}")
        print(f"  Latent FID: {latent_fid:.2f}")

        results.append({"frac": frac, "interp_t": interp_t,
                        "pixel_fid": pixel_fid, "latent_fid": latent_fid})

    # Summary
    print("\n" + "=" * 55)
    print(f"  Pixel  checkpoint: step {pixel_step}")
    print(f"  Latent checkpoint: step {latent_step}")
    print()
    print(f"  {'t_frac':>8}  {'t':>5}  {'Pixel FID':>10}  {'Latent FID':>10}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['frac']:8.4f}  {r['interp_t']:5d}  {r['pixel_fid']:10.2f}  {r['latent_fid']:10.2f}")
    print("=" * 55)

    results_path = os.path.join(args.output_dir, "fid_interpolation_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Pixel checkpoint:  {args.pixel_checkpoint}  (step {pixel_step})\n")
        f.write(f"Latent checkpoint: {args.latent_checkpoint}  (step {latent_step})\n")
        f.write(f"n_images={args.n_images}, n_interp={args.n_interp}\n")
        f.write(f"interp_t_fracs={args.interp_t_fracs}\n\n")
        f.write(f"{'t_frac':>8}  {'t':>5}  {'Pixel FID':>10}  {'Latent FID':>10}\n")
        for r in results:
            f.write(f"{r['frac']:8.4f}  {r['interp_t']:5d}  {r['pixel_fid']:10.4f}  {r['latent_fid']:10.4f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
