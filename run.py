"""
Entry point for DDPM training and sampling.

Usage:
    python run.py --mode train --total_steps 1300000
    python run.py --mode sample --resume checkpoints/ckpt_1300000.pt --n_samples 64
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="DDPM - Denoising Diffusion Probabilistic Models")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "denoise"])
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--total_steps", type=int, default=1_300_000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Train on first N images only (for overfit/validation testing)")

    # Sampling args
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of timestep columns in the denoising progression grid")
    parser.add_argument("--output_dir", type=str, default="samples")

    args = parser.parse_args()

    if args.mode == "train":
        from train import train
        train(
            dataset=args.dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size,
        )
    elif args.mode == "sample":
        if args.resume is None:
            parser.error("--resume is required for sampling")
        from sample import sample
        sample(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )
    elif args.mode == "denoise":
        if args.resume is None:
            parser.error("--resume is required for denoise mode")
        from sample import sample_progression
        sample_progression(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
